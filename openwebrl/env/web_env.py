import asyncio
import contextlib
from typing import Tuple, List, Dict, Optional, ByteString, Any
import os
import time

from playwright.async_api import async_playwright

from openwebrl.env.base_env import BaseEnv
from openwebrl.env.utils.js import find_clickable_elements_js_code
import logging

logger = logging.getLogger(__name__)

# Step hang guard. Playwright's page.evaluate has no timeout and
# set_default_timeout does not cover it, so a page whose main thread never
# yields (JS challenge loops, spinners) wedges the per-step a11ytree
# extraction, the env server's /step never returns, and the client's 300 s
# request timeout kills the process -> the task is scored env_step_error
# (~15% of episodes, 2026-09-05). With the guard, a wedged extraction degrades
# to an empty tree and a wedged action returns a failure message, so the
# episode continues instead.
_HANG_GUARD_A11Y_SECS = float(os.environ.get("SLIME_BROWSER_HANG_GUARD_A11Y_SECS", "20"))
_HANG_GUARD_ACTION_SECS = float(os.environ.get("SLIME_BROWSER_HANG_GUARD_ACTION_SECS", "60"))
_DEAD_POPUP_URLS = (":", "")  # uncommitted popup (navigation became a download)
logger.warning(
    "step hang guard: ON (a11y=%.0fs action=%.0fs)",
    _HANG_GUARD_A11Y_SECS, _HANG_GUARD_ACTION_SECS,
)


def _same_navigation_target(landed: str, requested: str) -> bool:
    """True if `landed` is the page we asked for, ignoring cosmetic URL differences.

    Used only to recover from Playwright's "interrupted by another navigation"
    error, where a site navigates to its own URL and aborts our pending goto.
    Compares scheme-insensitively on (host, path) with a trailing slash and a
    leading "www." normalised away, so "https://x.com" == "https://www.x.com/".
    Deliberately ignores query and fragment: the interrupting navigation is the
    site's own, and a task's start page is identified by its path.
    """
    from urllib.parse import urlparse

    def norm(u):
        try:
            p = urlparse(u)
        except Exception:
            return None
        host = (p.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host, (p.path or "/").rstrip("/") or "/"

    a, b = norm(landed), norm(requested)
    return a is not None and a == b


class WebEnv(BaseEnv):
    """
    Web environment for browser interaction using Playwright.
    Provides a interface for web automation tasks.
    """

    def __init__(self,
        # required parameters
        width: int,
        height: int,
        dpr: int,
        max_retries: int,
        wait_timeout: int,
        screenshot_timeout: Optional[int],
        start_url: str,
        tool_list: Optional[List[Dict[str, Any]]],
        policy: Optional[str],
        # optional parameters
        explicitly_allowed_ports: Optional[List[int]] = [],
    ) -> None:
        """
        Initialize the web environment (sync part).
        Note: You must call await setup() after creating an instance.
        
        Args:
            server_path: Path to the server, if any
            **kwargs: Additional web configuration parameters
                tool_list: should be consistent with OpenAI tool format.
        """
        super().__init__(server_path=None, platform="web")

        # Create context, set viewport size
        self.screen_size = (width, height)
        self.dpr = dpr
        self.max_retries = max_retries
        self.timeout = wait_timeout
        # Initial page loads for some sites (e.g. BBC/Cambridge) are
        # noticeably slower than regular interactive steps, so keep a more
        # forgiving timeout here without affecting the rest of the env.
        self.init_navigation_timeout = max(wait_timeout, 20000)
        self.screenshot_timeout = (
            screenshot_timeout if screenshot_timeout is not None else max(wait_timeout, 15000)
        )
        self.start_url = start_url
        self.css_width, self.css_height = int(self.screen_size[0] // self.dpr), int(
            self.screen_size[1] // self.dpr
        )

        self.tool_list = tool_list
        self.policy = policy

        self.browser_type = None
        self.browser = None
        self.context = None
        self.page = None

        self.browser_args = [
            "--disable-gpu",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-logging",
            "--ignore-certificate-errors",
            "--disable-dev-shm-usage",
            "--disable-application-cache",
            "--media-cache-size=0",
            "--disk-cache-size=0",
            "--log-level=3",
            "--silent",
            "--allow-running-insecure-content",
            "--disable-web-security",
            f"--explicitly-allowed-ports={','.join(str(p) for p in explicitly_allowed_ports)}",
        ]

        self.proxy_settings = None

    async def setup(self) -> None:
        """
        Async initialization of Playwright resources.
        Must be called after __init__.
        """
        self.playwright = await async_playwright().start()
        self.browser_type = self.playwright.chromium
        self.browser = await self.browser_type.launch(
            headless=True, args=self.browser_args, proxy=self.proxy_settings
        )

        await self._initialize_context(start_url=self.start_url)
        logger.info(
            f"Initializing WebEnv, browser type: {self.browser_type.name}, proxy settings: {self.proxy_settings}, viewport size: {self.screen_size}, DPR: {self.dpr}, timeout: {self.timeout}ms, screenshot_timeout: {self.screenshot_timeout}ms"
        )
        logger.info(f"WebEnv initialization successful!")

    async def reset(self) -> None:
        """
        Reset environment, create new context and navigate to the start URL.
        """
        # Set up browser env
        await self._initialize_context(start_url=self.start_url)

        observation = {
            "screenshot": await self.get_screenshot(),
            "a11ytree": await self.get_a11ytree(),
            "screen_size": await self.get_screen_size(), # width, height
            "all_tab_url": await self.get_all_tab_urls(),
            "active_tab_url": await self.get_active_tab_url(),
        }
        info = {
            "tool_list": self.tool_list,
            "policy": self.policy,
            "env_message": "Browser environment reset successfully.",
            "tool_response": [],
        }
        return observation, info

    async def _initialize_context(self,
        start_url: str = "about:blank",
    ) -> None:
        """
        Initialize or reinitialize browser context and page.

        Args:
            start_url: URL to navigate to after initialization
        """
        # If context already exists, close it first
        if self.context:
            try:
                await self.context.close()
            except Exception as e:
                logger.error(f"Error closing old context: {str(e)}")

        # Basic context configuration
        context_options = {
            "viewport": {"width": self.css_width, "height": self.css_height},
            "device_scale_factor": self.dpr,
            "is_mobile": False,
        }

        # Create new context and page
        self.context = await self.browser.new_context(**context_options)

        self.context.set_default_timeout(self.timeout)

        # If initial page exists, there might be multiple initial pages for complex tasks
        assert start_url
        start_urls = start_url.split(" |AND| ")
        for url in start_urls:
            page = await self.context.new_page()
            for attempt in range(self.max_retries):
                try:
                    await page.goto(url, timeout=self.init_navigation_timeout)
                    await page.wait_for_load_state("domcontentloaded")
                    break
                except Exception as e:
                    # A site that self-navigates on load (client-side reload, cookie
                    # bounce, trailing-slash redirect) INTERRUPTS the pending goto and
                    # Playwright raises -- even though the page is going exactly where
                    # we asked it to go. Retrying re-issues the identical goto, races
                    # identically, and burns every attempt, so the whole /reset fails.
                    #
                    # This was 72 of 80 navigation failures in obase-v6 (only 8 were
                    # real timeouts), concentrated on a few hosts in the webgym task
                    # set. It is rare on the om2w eval set (2 of 187 aborts), so the
                    # fix helps OPD without disturbing eval comparability.
                    #
                    # Recover ONLY when the page actually landed on the target. This
                    # branch runs only on the failure path, so it can turn an abort
                    # into a success but can never change a navigation that already
                    # succeeds.
                    if "interrupted by another navigation" in str(e):
                        try:
                            await page.wait_for_load_state(
                                "domcontentloaded", timeout=self.init_navigation_timeout
                            )
                            if _same_navigation_target(page.url, url):
                                logger.warning(
                                    f"Navigation to {url} was interrupted by the page navigating "
                                    f"to itself; landed on {page.url}, treating as success."
                                )
                                break
                        except Exception:
                            pass  # fall through to the normal retry/raise below
                    if attempt < self.max_retries - 1:
                        logger.warning(f"Failed to navigate to {url} (attempt {attempt + 1}/{self.max_retries}): {e}. Retrying...")
                        await asyncio.sleep(2)
                    else:
                        raise EnvironmentError(f"Failed to navigate to {url} after {self.max_retries} attempts: {e}")

        # Set first page as current starting page
        self.page = self.context.pages[0]
        await self.page.bring_to_front()

        # Register the page listener last -- this listener is used to automatically
        # switch Page handles. setup_dialog_interceptor() is scoped to the current
        # self.page object (captured by value at call time) rather than to the
        # context, so it is attached here too: pages opened directly via
        # context.new_page() above (as opposed to browser-triggered popups/new tabs,
        # which setup_global_page_listener's own _handle_new_page callback already
        # re-attaches this to) never fire the context's "page" event.
        self.setup_global_page_listener()
        self.setup_dialog_interceptor()
        await asyncio.sleep(2)

    async def exit(self) -> None:
        """
        Close all Playwright resources and exit.
        """
        if self.page and not self.page.is_closed():
            await self.page.close()

        if self.context:
            await self.context.close()

        if self.browser:
            await self.browser.close()

        if self.playwright:
            await self.playwright.stop()

    async def find_all_clickable_elements(self) -> List[Dict[str, Any]]:
        """
        Get all clickable elements on the page.

        Returns:
            List of dictionaries containing information about clickable elements
        """
        try:
            all_clickable_elements_info = await self.page.evaluate(
                find_clickable_elements_js_code
            )

            # Normalize returned coordinates[0~1]
            return [
                {
                    **ele,
                    "bbox": [
                        (
                            coord / self.css_width
                            if i % 2 == 0
                            else coord / self.css_height
                        )
                        for i, coord in enumerate(ele["bbox"])
                    ],
                }
                for ele in all_clickable_elements_info
            ]

        except Exception as e:
            print(str(e))
            return []

    async def get_a11ytree(self) -> List[Dict[str, Any]]:
        """
        Get accessibility tree of clickable elements.

        Returns:
            List of dictionaries containing information about clickable elements
        """
        try:
            return await asyncio.wait_for(
                self.find_all_clickable_elements(), timeout=_HANG_GUARD_A11Y_SECS
            )
        except asyncio.TimeoutError:
            logger.warning(
                "hang guard: a11ytree extraction exceeded %.0fs; returning empty tree",
                _HANG_GUARD_A11Y_SECS,
            )
            return []

    async def get_screen_size(self) -> Tuple[int, int]:
        """
        Get current screen size.

        Returns:
            Tuple of (width, height) in pixels
        """
        viewport = self.page.viewport_size
        return int(viewport["width"] * self.dpr), int(viewport["height"] * self.dpr)

    async def get_screenshot(self) -> ByteString:
        """
        Take a screenshot of the current page.

        Returns:
            Screenshot as bytes
        """
        try:
            screenshot_bytes = await self.page.screenshot(timeout=self.screenshot_timeout)
        except Exception as exc:
            # Playwright's screenshot blocks on document.fonts.ready ("waiting for fonts to
            # load...") and times out on pages whose font loading never settles -- ~10% of
            # env_step_error aborts, and 100% of the 2026-09-05 smoke aborts. The hang guard
            # falls back to a raw CDP capture, which does not wait for fonts.
            logger.warning("hang guard: page.screenshot failed (%s); falling back to CDP capture", type(exc).__name__)
            if await self._recover_from_dead_popup():
                return await self.get_screenshot()
            cdp = await self.page.context.new_cdp_session(self.page)
            try:
                result = await asyncio.wait_for(
                    cdp.send("Page.captureScreenshot", {"format": "png"}), timeout=_HANG_GUARD_A11Y_SECS
                )
            finally:
                with contextlib.suppress(Exception):
                    await cdp.detach()
            import base64 as _b64
            screenshot_bytes = _b64.b64decode(result["data"])
        return screenshot_bytes

    async def get_all_tab_urls(self) -> List[Dict[str, Any]]:
        """
        Get URLs of all opened tabs in the current browser context.

        Returns:
            List of dictionaries containing tab index, URL, title, and active flag, e.g.:
            [
                {"index": 0, "url": "https://example.com", "title": "Example", "active": True},
                {"index": 1, "url": "https://google.com", "title": "Google", "active": False}
            ]
        """
        if not self.context:
            return []
        
        tabs_info = []
        for i, page in enumerate(self.context.pages):
            try:
                title = await page.title()
            except Exception:
                title = ""
            if len(title) > 50:
                title = title[:50]
            tabs_info.append({
                "index": i,
                "url": page.url,
                "title": title,
                "active": page == self.page,
            })
        return tabs_info

    async def get_active_tab_url(self) -> Optional[str]:
        """
        Get the URL of the currently active tab.

        Returns:
            URL string of the active tab, or None if no page is active
        """
        if not self.page:
            return None
        return self.page.url

    @staticmethod
    def _get_element_signature(element: Dict[str, Any]) -> tuple:
        """Build a compact semantic signature for an a11ytree element."""
        tag = element.get("tag", "")
        etype = element.get("type", "") or ""
        text = (
            element.get("ariaLabel")
            or element.get("textContent")
            or element.get("innerText")
            or ""
        ).strip()
        if len(text) > 30:
            text = text[:30]
        return (tag, etype, text)

    def _diff_a11ytree(self, before: list, after: list) -> str:
        """Compare two a11ytree snapshots and return a short summary of changes."""
        before_sigs = {}
        for el in before:
            sig = self._get_element_signature(el)
            before_sigs[sig] = before_sigs.get(sig, 0) + 1

        after_sigs = {}
        for el in after:
            sig = self._get_element_signature(el)
            after_sigs[sig] = after_sigs.get(sig, 0) + 1

        added = []
        for sig, count in after_sigs.items():
            diff = count - before_sigs.get(sig, 0)
            for _ in range(diff):
                added.append(sig)

        removed = []
        for sig, count in before_sigs.items():
            diff = count - after_sigs.get(sig, 0)
            for _ in range(diff):
                removed.append(sig)

        if not added and not removed:
            return ""

        def _fmt(sig: tuple) -> str:
            tag, etype, text = sig
            desc = f"<{tag}>"
            if etype:
                desc += f" type={etype}"
            if text:
                desc += f" '{text}'"
            return desc

        parts = []
        if added:
            shown = [_fmt(sig) for sig in added[:5]]
            desc = f"{len(added)} element(s) appeared [{', '.join(shown)}]"
            if len(added) > 5:
                desc += f" and {len(added) - 5} more"
            parts.append(desc)
        if removed:
            shown = [_fmt(sig) for sig in removed[:5]]
            desc = f"{len(removed)} element(s) disappeared [{', '.join(shown)}]"
            if len(removed) > 5:
                desc += f" and {len(removed) - 5} more"
            parts.append(desc)

        return "DOM changes: " + "; ".join(parts) + "."

    async def _recover_from_dead_popup(self) -> bool:
        """
        Hang guard: a popup whose navigation turned into a download (e.g. a target=_blank
        PDF link -- the headless shell has no PDF viewer) never commits: its url stays ':'
        and both page.screenshot and CDP Page.captureScreenshot hang on it forever
        (verified 2026-09-05, job 39638160). Close it and return to the previous page.
        """
        page = self.page
        if page is None or page.is_closed() or page.url not in _DEAD_POPUP_URLS:
            return False
        others = [
            p for p in self.context.pages
            if p is not page and not p.is_closed() and p.url not in _DEAD_POPUP_URLS
        ]
        if not others:
            return False
        logger.warning(
            "hang guard: download-only popup (url=%r); closing it and returning to %s", page.url, others[-1].url
        )
        with contextlib.suppress(Exception):
            await asyncio.wait_for(page.close(), timeout=5)
        self.page = others[-1]
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self.page.bring_to_front(), timeout=5)
        self._popup_note = (
            "Note: the last action opened a file download in a new tab that cannot be displayed; "
            "that tab was closed and the previous page is shown."
        )
        return True

    def setup_global_page_listener(self) -> None:
        """
        Set up global page listener to automatically switch to new pages and close old ones.
        Critical: Ensures each operation occurs on the current page.
        """

        async def _handle_new_page(page):
            logger.info(f"New page detected: {page.url}")
            old_page = self.page
            self.page = page
            try:
                await asyncio.wait_for(
                    self.page.wait_for_load_state("domcontentloaded"), timeout=_HANG_GUARD_A11Y_SECS
                )
            except Exception as exc:
                logger.warning("hang guard: new page load wait failed (%s)", type(exc).__name__)
            if await self._recover_from_dead_popup():
                return
            self.setup_dialog_interceptor()
            logger.info(f"Switched to new page: {self.page.url}")

        # Add page listener
        self.context.on("page", _handle_new_page)

    def setup_dialog_interceptor(self) -> None:
        """
        Set up dialog interceptor to replace native popups with DOM popups.
        """
        if not self.page:
            return

        # Capture current page to avoid using stale references if self.page changes during navigation
        target_page = self.page

        # Create a new dialog intercept handler
        def intercept_dialog(dialog):
            try:
                # Immediately handle native dialog - accept all dialogs by default
                dialog_type = dialog.type
                dialog_message = dialog.message
                dialog_default_value = (
                    dialog.default_value if dialog_type == "prompt" else ""
                )

                # Immediately handle the native dialog
                if dialog_type == "prompt":
                    asyncio.create_task(dialog.accept(dialog_default_value))
                else:
                    asyncio.create_task(dialog.accept())

                # Record dialog information
                dialog_id = f"mock-dialog-{int(time.time() * 1000)}"

                # Inject mock dialog into DOM
                # Safety check: ensure page is still valid before injection
                if target_page.is_closed():
                    return

                asyncio.create_task(target_page.evaluate(
                    """
                    ({ type, message, defaultValue, dialogId }) => {
                        // Clean up existing mock popups
                        const existingMock = document.getElementById('mock-dialog-container');
                        if (existingMock) existingMock.remove();

                        // Create container
                        const container = document.createElement('div');
                        container.id = 'mock-dialog-container';
                        container.style = `
                            position: fixed;
                            top: 0;
                            left: 0;
                            right: 0;
                            bottom: 0;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            z-index: 99999;
                            font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                        `;

                        // Create overlay
                        const overlay = document.createElement('div');
                        overlay.style = `
                            position: absolute;
                            top: 0;
                            left: 0;
                            right: 0;
                            bottom: 0;
                            background: rgba(0, 0, 0, 0.5);
                            z-index: 1;
                        `;

                        // Create dialog
                        const dialog = document.createElement('div');
                        dialog.id = dialogId;
                        dialog.style = `
                            background: white;
                            border-radius: 8px;
                            box-shadow: 0 4px 23px 0 rgba(0, 0, 0, 0.2);
                            padding: 20px;
                            min-width: 320px;
                            max-width: 90vw;
                            position: relative;
                            z-index: 2;
                            overflow: hidden;
                        `;

                        // Message text
                        const messageElement = document.createElement('div');
                        messageElement.style = `
                            margin-bottom: 20px;
                            font-size: 14px;
                            color: #333;
                            line-height: 1.5;
                            word-break: break-word;
                        `;
                        messageElement.textContent = message;

                        // Button container
                        const buttonsContainer = document.createElement('div');
                        buttonsContainer.style = `
                            display: flex;
                            justify-content: flex-end;
                            gap: 10px;
                        `;

                        let inputElement = null;
                        if (type === 'prompt') {
                            // Add input field
                            inputElement = document.createElement('input');
                            inputElement.type = 'text';
                            inputElement.value = defaultValue || '';
                            inputElement.style = `
                                width: 100%;
                                padding: 10px;
                                border: 1px solid #ddd;
                                border-radius: 4px;
                                margin-bottom: 15px;
                                font-size: 14px;
                            `;
                        }

                        // OK button
                        const okButton = document.createElement('button');
                        okButton.className = 'ok-button';
                        okButton.textContent = 'OK';
                        okButton.style = `
                            padding: 8px 16px;
                            background: #0d6efd;
                            border: none;
                            border-radius: 4px;
                            cursor: pointer;
                            font-size: 14px;
                            color: white;
                        `;
                        okButton.addEventListener('mouseover', () => {
                            okButton.style.background = '#0b5ed7';
                        });
                        okButton.addEventListener('mouseout', () => {
                            okButton.style.background = '#0d6efd';
                        });
                        okButton.addEventListener('click', () => {
                            // Close mock dialog
                            container.remove();
                        });
                        buttonsContainer.appendChild(okButton);

                        // Add elements to dialog
                        dialog.appendChild(messageElement);
                        if (inputElement) {
                            dialog.appendChild(inputElement);
                        }
                        dialog.appendChild(buttonsContainer);

                        // Add all elements to container
                        container.appendChild(overlay);
                        container.appendChild(dialog);

                        // Add to document
                        document.body.appendChild(container);

                        // Support ESC key and background click to close
                        const handleKeyDown = (e) => {
                            if (e.key === 'Escape') {
                                container.remove();
                                document.removeEventListener('keydown', handleKeyDown);
                            }
                        };
                        document.addEventListener('keydown', handleKeyDown);

                        // Close on background click
                        overlay.addEventListener('click', () => {
                            container.remove();
                        });
                    }
                """,
                    {
                        "type": dialog_type,
                        "message": dialog_message,
                        "defaultValue": dialog_default_value,
                        "dialogId": dialog_id,
                    },
                ))

                # Log dialog information
                logger.info(
                    f"Dialog intercepted and handled: {dialog_type} - {dialog_message}"
                )

            except Exception as e:
                # If the error is about execution context being destroyed, it's likely a navigation 
                # occurred right after dialog.accept(). This is expected and can be ignored.
                if "Execution context was destroyed" in str(e):
                    logger.debug(f"Error handling dialog: Context destroyed during dialog handling (likely navigation): {str(e)}")
                else:
                    logger.error(f"Error handling dialog: {str(e)}")
                
                # If error during handling, try to dismiss dialog (if it wasn't already)
                try:
                    asyncio.create_task(dialog.dismiss())
                except:
                    pass

        target_page.on("dialog", intercept_dialog)

    async def execute_single_action(self, action: Dict[str, Any]) -> tuple[bool, str]:
        """
        Execute a single action based on action type.

        Args:
            action: Dictionary containing action type and args

        Returns:
            tuple[bool, str]: (success flag, result message)
        """
        action_type = action["name"]
        parameters = action.get("args", {})
        try:
            # Action type mapping table
            action_handlers = {
                "done": self._execute_done,
                "goto_url": self._execute_goto_url,
                "go_back": self._execute_go_back,
                "new_tab": self._execute_new_tab,
                "switch_tab": self._execute_switch_tab,
                "close_tab": self._execute_close_tab,
            }

            # Get corresponding handler and execute
            handler = action_handlers.get(action_type)
            if handler:
                flag, exe_msg = await handler(parameters)
                return flag, exe_msg
            else:
                return False, f"Error: Action {action_type} is not supported by the environment!"
        except Exception as e:
            return False, f"Error occurred when parsing actions in the environment: {e}"

    async def _execute_done(self, parameters: Dict[str, Any]) -> tuple[bool, str]:
        """
        Signal task completion and provide the final answer.

        Args:
            parameters: Dictionary containing:
                - response (str): Final answer or task summary

        Returns:
            tuple[bool, str]: (success flag, result message)
        """
        response = parameters.get("response", "")
        return True, f"Succeed: task marked as done. Response: {response}"

    async def _execute_go_back(self, parameters: Dict[str, Any]) -> tuple[bool, str]:
        """
        Navigate back to the previous page in browser history.

        Args:
            parameters: Dictionary (unused)

        Returns:
            tuple[bool, str]: (success flag, result message)
        """
        try:
            await self.page.go_back()
            try:
                await self.page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass
            await self.page.wait_for_timeout(1000)
            return True, "Succeed: `go_back` executed."
        except Exception as e:
            return False, f"Failed: `go_back` execution failed: {e}"

    async def _execute_goto_url(self, parameters: Dict[str, Any]) -> tuple[bool, str]:
        """
        Navigate the browser to a specific URL.

        Args:
            parameters: Dictionary containing:
                - url (str): The URL to navigate to (required)

        Returns:
            tuple[bool, str]: (success flag, result message)
        """
        try:
            url = parameters.get("url")
            if not url:
                return False, "Failed: `goto_url` requires a non-empty `url`."

            await self.page.goto(url, timeout=self.timeout)
            try:
                await self.page.wait_for_load_state("domcontentloaded", timeout=self.timeout)
            except Exception:
                pass
            await self.page.wait_for_timeout(1000)
            return True, f"Succeed: `goto_url` navigated to {url}."
        except Exception as e:
            return False, f"Failed: `goto_url` execution failed: {e}"

    async def _execute_new_tab(self, parameters: Dict[str, Any]) -> tuple[bool, str]:
        """
        Open a new blank browser tab.

        Returns:
            tuple[bool, str]: (success flag, result message)
        """
        try:
            new_page = await self.context.new_page()
            self.page = new_page
            self.setup_dialog_interceptor()
            await self.page.goto("about:blank")

            await self.page.wait_for_timeout(500)
            tab_index = self.context.pages.index(self.page)
            return True, f"Succeed: `new_tab` opened blank tab (tab {tab_index})."
        except Exception as e:
            return False, f"Failed: `new_tab` execution failed: {e}"

    async def _execute_switch_tab(self, parameters: Dict[str, Any]) -> tuple[bool, str]:
        """
        Switch to a different browser tab by index.

        Args:
            parameters: Dictionary containing:
                - tab_index (int): 0-based index of the tab to switch to (required)

        Returns:
            tuple[bool, str]: (success flag, result message)
        """
        try:
            tab_index = parameters.get("tab_index")
            if tab_index is None:
                return False, "Failed: `switch_tab` requires a `tab_index`."

            pages = self.context.pages
            if tab_index < 0 or tab_index >= len(pages):
                return False, f"Failed: `switch_tab` tab_index {tab_index} out of range (0-{len(pages) - 1})."

            self.page = pages[tab_index]
            await self.page.bring_to_front()
            await self.page.wait_for_timeout(500)
            return True, f"Succeed: `switch_tab` switched to tab {tab_index} ({self.page.url})."
        except Exception as e:
            return False, f"Failed: `switch_tab` execution failed: {e}"

    async def _execute_close_tab(self, parameters: Dict[str, Any]) -> tuple[bool, str]:
        """
        Close the current browser tab and switch to the nearest remaining tab.

        Args:
            parameters: Dictionary (unused)

        Returns:
            tuple[bool, str]: (success flag, result message)
        """
        try:
            pages = self.context.pages
            if len(pages) <= 1:
                return False, "Failed: `close_tab` cannot close the last remaining tab."

            current_index = pages.index(self.page)
            await self.page.close()

            remaining_pages = self.context.pages
            new_index = min(current_index, len(remaining_pages) - 1)
            self.page = remaining_pages[new_index]
            await self.page.bring_to_front()
            await self.page.wait_for_timeout(500)
            return True, f"Succeed: `close_tab` closed tab {current_index}, now on tab {new_index}."
        except Exception as e:
            return False, f"Failed: `close_tab` execution failed: {e}"

    async def step(self, action_list: list[dict]):
        """
        Execute environment step with given actions (async version).

        Args:
            action_list: List of actions to execute
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        self.action_history.append(action_list)
        # For now, return simple gym-style outputs
        # You can customize these based on your needs
        observation = {}
        reward = 0.0
        terminated = False
        truncated = False
        info = {
            "tool_list": self.tool_list,
            "policy": self.policy,
            "env_message": "Start to execute actions....",
            "tool_responses": [],
        }

        try:
            pre_a11ytree = await self.get_a11ytree()

            for action in action_list:
                try:
                    flag, msg = await asyncio.wait_for(
                        self.execute_single_action(action), timeout=_HANG_GUARD_ACTION_SECS
                    )
                except asyncio.TimeoutError:
                    flag, msg = False, (
                        f"Failed: `{action.get('name')}` timed out after "
                        f"{_HANG_GUARD_ACTION_SECS:.0f}s (page unresponsive)."
                    )
                    logger.warning("hang guard: action %s timed out", action.get("name"))
                info["tool_responses"].append({
                    "tool_name": action["name"],
                    "tool_response": msg
                })
                terminated = True if (action["name"] == "done" and flag) else False
            
            await self._recover_from_dead_popup()
            _note = getattr(self, "_popup_note", None)
            if _note:
                info["env_message"] += " " + _note
                self._popup_note = None

            post_a11ytree = await self.get_a11ytree()
            dom_diff = self._diff_a11ytree(pre_a11ytree, post_a11ytree)
            if dom_diff:
                info["env_message"] += " " + dom_diff

            # Get new observation
            observation = {
                "screenshot": await self.get_screenshot(),
                "a11ytree": post_a11ytree,
                "screen_size": await self.get_screen_size(), # width, height
                "all_tab_url": await self.get_all_tab_urls(),
                "active_tab_url": await self.get_active_tab_url(),
            }
                        
        except Exception as e:
            info["env_message"] = f"Error during env.step(): {type(e).__name__}: {str(e)}"
        
        return observation, reward, terminated, truncated, info
