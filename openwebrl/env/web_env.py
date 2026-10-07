import shutil
import tempfile
import asyncio
import contextlib
from typing import Tuple, List, Union, Dict, Optional, ByteString, Any
import os
import time

from playwright.async_api import async_playwright

from openwebrl.env.base_env import BaseEnv
from openwebrl.feedback_utils import (
    DEFAULT_BROWSER_ACTUAL_VALUE_MAX_CHARS,
    truncate_feedback_text,
)
from openwebrl.env.utils.js import find_clickable_elements_js_code
from openwebrl.env import molmoweb_feedback
from openwebrl.env.utils.keyboard import validate_and_correct_key
import logging

logger = logging.getLogger(__name__)

# Step hang guard (inert unless SLIME_BROWSER_STEP_HANG_GUARD=1). Playwright's
# page.evaluate has no timeout and set_default_timeout does not cover it, so a
# page whose main thread never yields (JS challenge loops, spinners) wedges the
# per-step a11ytree extraction, the env server's /step never returns, and the
# client's 300 s request timeout kills the process -> the task is scored
# env_step_error (~15% of episodes, 2026-09-05). With the guard on, a wedged
# extraction degrades to an empty tree, a wedged action returns a failure
# message, and wait(seconds) is clamped, so the episode continues instead.
_HANG_GUARD = os.environ.get("SLIME_BROWSER_STEP_HANG_GUARD", "0") == "1"
_HANG_GUARD_A11Y_SECS = float(os.environ.get("SLIME_BROWSER_HANG_GUARD_A11Y_SECS", "20"))
_HANG_GUARD_ACTION_SECS = float(os.environ.get("SLIME_BROWSER_HANG_GUARD_ACTION_SECS", "60"))
_HANG_GUARD_WAIT_MAX_SECS = int(os.environ.get("SLIME_BROWSER_HANG_GUARD_WAIT_MAX_SECS", "10"))
_DEAD_POPUP_URLS = (":", "")  # uncommitted popup (navigation became a download)
logger.warning(
    "step hang guard: %s (a11y=%.0fs action=%.0fs wait_max=%ds)",
    "ON" if _HANG_GUARD else "OFF", _HANG_GUARD_A11Y_SECS, _HANG_GUARD_ACTION_SECS, _HANG_GUARD_WAIT_MAX_SECS,
)

_UPSCALE_FACTOR = None


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


def _maybe_upscale_screenshot(png_bytes):
    """Interpolation-only upscale of the captured screenshot.

    Control for the dpr-2 arm: dpr renders the page at a finer pixel grid, so it
    adds real detail, while this resamples an already-captured baseline frame to
    the same pixel count and therefore adds NO information. Comparing the two
    separates "more visual detail helps" from "more vision tokens helps".

    Deliberately operates only on the encoded image. The viewport, dpr and
    screen_size are untouched, so css_width is unchanged and the 0-1000 action
    coordinate mapping in execute_single_action still resolves correctly.
    """
    global _UPSCALE_FACTOR
    if _UPSCALE_FACTOR is None:
        raw = os.environ.get("SLIME_BROWSER_UPSCALE_FACTOR", "1")
        try:
            _UPSCALE_FACTOR = float(raw)
        except ValueError:
            raise ValueError(
                f"SLIME_BROWSER_UPSCALE_FACTOR must be a number, got {raw!r}."
            )
        if _UPSCALE_FACTOR != 1:
            logger.info("Screenshot interpolation upscale active: x%s", _UPSCALE_FACTOR)
    if _UPSCALE_FACTOR == 1:
        return png_bytes

    from io import BytesIO
    from PIL import Image

    img = Image.open(BytesIO(png_bytes))
    target = (
        int(round(img.width * _UPSCALE_FACTOR)),
        int(round(img.height * _UPSCALE_FACTOR)),
    )
    out = BytesIO()
    img.resize(target, Image.BICUBIC).save(out, format="PNG")
    return out.getvalue()


_CHANGE_MAP = None


def _maybe_change_map(png_bytes, prev_png_bytes):
    """Change-map overlay (inert unless SLIME_BROWSER_CHANGE_MAP=1).

    Draws a red rectangle around the region of the page that changed since the
    previous capture, so the agent can SEE whether its last action did anything and
    where. Zero extra vision tokens; pixels only, coordinates untouched. Inspired by
    ExeVRM's temporal token pruning (correctness lives in local transitions).
    Diff is computed on a 1/8-scale greyscale; regions under 0.2% of the page are
    ignored (cursor blink, spinners); a >90% change (navigation) is not localised.
    Returns (bytes_to_serve, raw_bytes_to_remember).
    """
    global _CHANGE_MAP
    if _CHANGE_MAP is None:
        _CHANGE_MAP = os.environ.get("SLIME_BROWSER_CHANGE_MAP", "0") == "1"
        if _CHANGE_MAP:
            logger.info("Change-map overlay active")
    if not _CHANGE_MAP or prev_png_bytes is None:
        return png_bytes, png_bytes
    from io import BytesIO
    import numpy as np
    from PIL import Image, ImageDraw
    cur = Image.open(BytesIO(png_bytes)).convert("RGB")
    prev = Image.open(BytesIO(prev_png_bytes)).convert("RGB")
    if prev.size != cur.size:
        return png_bytes, png_bytes
    s = 8
    a = np.asarray(cur.resize((cur.width // s, cur.height // s)).convert("L"), dtype=np.int16)
    b = np.asarray(prev.resize((prev.width // s, prev.height // s)).convert("L"), dtype=np.int16)
    changed = np.abs(a - b) > 24
    if changed.mean() < 0.002:
        return png_bytes, png_bytes
    ys, xs = np.where(changed)
    x0, x1 = int(xs.min()) * s, int(xs.max() + 1) * s
    y0, y1 = int(ys.min()) * s, int(ys.max() + 1) * s
    if (x1 - x0) * (y1 - y0) > 0.9 * cur.width * cur.height:
        return png_bytes, png_bytes
    out = cur.copy()
    ImageDraw.Draw(out).rectangle([x0, y0, x1 - 1, y1 - 1], outline=(255, 0, 0), width=4)
    buf = BytesIO(); out.save(buf, format="PNG")
    return buf.getvalue(), png_bytes


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
        resize_output_coords: bool,
        resize_scale: int,
        image_patch_size: int,
        tool_list: Optional[List[Dict[str, Any]]],
        policy: Optional[str],
        # optional parameters
        storage_state: Optional[dict] = None,
        explicitly_allowed_ports: Optional[List[int]] = [],
        web_proxy: Optional[str] = None,
        # Opt-in cloud browser backend (default "local" preserves existing
        # behavior exactly). "browserbase" connects to a Browserbase cloud
        # session instead of launching local Chromium -- stealth fingerprint,
        # rotating residential proxies, and automatic CAPTCHA solving, aimed at
        # the CAPTCHA/IP-reputation blocking documented in the Same Model
        # Family ablation (this cluster's shared egress IP getting walled by
        # Google/DuckDuckGo/allrecipes.com regardless of agent behavior).
        # Modeled on molmoweb/utils/envs/browser_env.py's BrowserbaseEnv.
        browser_backend: str = "local",
        browserbase_api_key: Optional[str] = None,
        browserbase_project_id: Optional[str] = None,
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

        self.auth_info = storage_state

        # Opt-in MolmoWeb-vocabulary feedback (see molmoweb_feedback.py). Off by
        # default: every other checkpoint trained on this harness's own strings.
        # The axtree is extracted once per step() and cached here, because
        # extraction marks and unmarks the live DOM and is far too costly to run
        # per action.
        self._molmoweb_feedback = molmoweb_feedback.enabled()
        self._molmoweb_axtree = None
        self._molmoweb_extra = None

        # Action Execution - point_2d format parameters
        self.smart_resize_height = None
        self.smart_resize_width = None
        if resize_output_coords:
            if resize_scale <= 0:
                from qwen_vl_utils import smart_resize
                self.smart_resize_height, self.smart_resize_width = smart_resize(
                    self.height,
                    self.width,
                    factor=image_patch_size,
                )
            else:
                self.smart_resize_height, self.smart_resize_width = resize_scale, resize_scale
        
                self.playwright = None

        self.browser_type = None
        self.browser = None
        self.should_record = False
        self.temp_video_dir = None
        self.context = None
        self.page = None

        if browser_backend not in ("local", "browserbase"):
            raise ValueError(f"Unsupported browser_backend={browser_backend!r}; expected 'local' or 'browserbase'.")
        self.browser_backend = browser_backend
        self.browserbase_api_key = browserbase_api_key or os.environ.get("BROWSERBASE_API_KEY")
        self.browserbase_project_id = browserbase_project_id or os.environ.get("BROWSERBASE_PROJECT_ID")
        self.bb = None
        self.bb_session = None

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

        # Proxy configuration
        if isinstance(web_proxy, str):
            proxy_username = web_proxy.split("//")[1].split(":")[0]
            proxy_password = web_proxy.split("//")[1].split(":")[1].split("@")[0]
            proxy_server = "http://" + web_proxy.split("//")[1].split("@")[1]

            # Create proxy settings
            self.proxy_settings = {
                "server": proxy_server,
                "username": proxy_username,
                "password": proxy_password,
            }
        else:
            self.proxy_settings = None

    async def setup(self) -> None:
        """
        Async initialization of Playwright resources.
        Must be called after __init__.
        """
        self.playwright = await async_playwright().start()
        self.browser_type = self.playwright.chromium
        if self.browser_backend == "browserbase":
            await self._launch_browserbase()
        else:
            self.browser = await self.browser_type.launch(
                headless=True, args=self.browser_args, proxy=self.proxy_settings
            )

        await self._initialize_context(
            enable_recording=self.should_record, 
            start_url=self.start_url, 
            auth_info=self.auth_info
        )
        logger.info(
            f"Initializing WebEnv, browser type: {self.browser_type.name}, proxy settings: {self.proxy_settings}, viewport size: {self.screen_size}, DPR: {self.dpr}, timeout: {self.timeout}ms, screenshot_timeout: {self.screenshot_timeout}ms"
        )
        logger.info(f"WebEnv initialization successful!")

    async def _launch_browserbase(self) -> None:
        """
        Connect to a Browserbase cloud session instead of launching local
        Chromium. Mirrors molmoweb/utils/envs/browser_env.py's
        BrowserbaseEnv._launch() (sync there; async here to match this
        class's Playwright API).
        """
        from browserbase import Browserbase

        if not self.browserbase_api_key or not self.browserbase_project_id:
            raise ValueError(
                "browser_backend='browserbase' requires browserbase_api_key and "
                "browserbase_project_id (pass explicitly or set BROWSERBASE_API_KEY / "
                "BROWSERBASE_PROJECT_ID)."
            )

        self.bb = Browserbase(api_key=self.browserbase_api_key)
        self.bb_session = self.bb.sessions.create(
            project_id=self.browserbase_project_id,
            proxies=True,
            browser_settings={
                "advanced_stealth": True,
                "viewport": {"width": self.css_width, "height": self.css_height},
            },
        )
        logger.info(f"Browserbase session created: {self.bb_session.id}")

        cdp_url = f"wss://connect.browserbase.com?sessionId={self.bb_session.id}&apiKey={self.browserbase_api_key}"
        self.browser = await self.browser_type.connect_over_cdp(cdp_url)

    async def start_recording(self) -> None:
        """
        Mark that video recording should begin (actual recording starts on next reset)
        """
        self.should_record = True
        logger.info("Recording flag set, recording will start on next reset")

    async def end_recording(self, path: str) -> None:
        """
        End video recording, save file, and remove current window context

        Args:
            path: Path where to save the recording
        """
        if not self.should_record:
            logger.info("No active recording in progress")
            return None

        try:
            # Close context
            await self.context.close()
            self.context = None
            self.should_record = False

            # Wait for video file to complete writing
            max_wait = 10  # Maximum 10 seconds wait
            start_time = time.time()

            # Wait for video file to appear
            latest_video_path = None
            while time.time() - start_time < max_wait:
                video_files = [
                    f
                    for f in os.listdir(self.temp_video_dir.name)
                    if f.endswith(".webm")
                ]
                if video_files:
                    # Sort by modification time, get most recent video file
                    video_files.sort(
                        key=lambda f: os.path.getmtime(
                            os.path.join(self.temp_video_dir.name, f)
                        ),
                        reverse=True,
                    )
                    latest_video_path = os.path.join(
                        self.temp_video_dir.name, video_files[0]
                    )
                    if os.path.getsize(latest_video_path) > 0:
                        break
                await asyncio.sleep(0.5)

            # Ensure destination path's directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Move the video to final destination
            shutil.move(latest_video_path, path)
            logger.info(f"Video moved to {path}")

        except Exception as e:
            logger.error(f"Error ending video recording: {str(e)}")

    async def reset(self,
        url: Optional[str] = None,
        auth_info: Optional[dict] = None,
    ) -> None:
        """
        Reset environment, create new context and navigate to specified URL.
        """
        self._cm_prev = None  # change-map: never diff across episodes
        if not url:
            url = self.start_url
        if not auth_info:
            auth_info = self.auth_info
        # Set up browser env
        await self._initialize_context(
            enable_recording=self.should_record, start_url=url, auth_info=auth_info
        )

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
            # None for browser_backend="local". Surfaced so callers can persist it
            # into saved trajectories -- the only way to confirm from shared storage
            # that a Browserbase session was actually created for a given rollout,
            # since WebEnv's own logs live in the env_server subprocess's node-local
            # /tmp (not reachable/readable from off that specific compute node).
            "bb_session_id": self.bb_session.id if self.bb_session else None,
        }
        return observation, info

    def evaluate_webarena(self, **kwargs) -> float:
        """
        Evaluate WebArena tasks using the appropriate evaluator.

        Args:
            **kwargs: Keyword arguments containing:
                actions: List of actions performed by the agent

        Returns:
            float: Evaluation score between 0 and 1
        """
        action_list = kwargs["actions"]
        evaluator = webarena_evaluator_router(self.task_config)
        score = evaluator(
            action_list=action_list, task_config=self.task_config, page=self.page
        )
        return score

    def evaluate(self, **kwargs) -> float:
        """
        Evaluate the current trajectory and return a score based on the specified benchmark.

        Args:
            **kwargs: Keyword arguments containing:
                benchmark: The benchmark name, default is 'vab_webarena_lite', can also be 'demo' or 'webvoyager_golden'
                actions: The list of actions performed by the agent

        Returns:
            float: Evaluation score between 0 and 1
        """
        evaluate_dict = {
            "vab_webarena_lite": self.evaluate_webarena,
            "webvoyager_golden": self.evaluate_webarena,
            "demo": self.evaluate_demo,
        }
        score = evaluate_dict[kwargs["benchmark"]](**kwargs)
        return score

    async def _initialize_context(self,
        enable_recording: bool = False,
        start_url: str = "about:blank",
        auth_info: Optional[dict] = None,
    ) -> None:
        """
        Initialize or reinitialize browser context and page.

        Args:
            enable_recording: Whether to enable video recording
            start_url: URL to navigate to after initialization
            state_path: Path to file for authentication. Example: ./auth/reddit_state.json
        """
        is_new_context = True
        if self.browser_backend == "browserbase" and self.context is not None:
            # Browserbase provisions its own default context/page with its stealth
            # fingerprint already baked in; closing and recreating it (as the local
            # path does below) would silently discard that fingerprint on every
            # reset. Reuse the same context across the env's lifetime instead --
            # just close out pages left over from a previous episode.
            is_new_context = False
            for page in list(self.context.pages):
                try:
                    await page.close()
                except Exception as e:
                    logger.error(f"Error closing old page: {str(e)}")
        else:
            # If context already exists, close it first
            if self.context:
                try:
                    await self.context.close()
                except Exception as e:
                    logger.error(f"Error closing old context: {str(e)}")

            if self.browser_backend == "browserbase":
                # First-ever initialization on this Browserbase session: use the
                # session's provisioned default context (see module note above),
                # not browser.new_context(), which would bypass its fingerprinting.
                self.context = self.browser.contexts[0]
            else:
                # Basic context configuration
                context_options = {
                    "viewport": {"width": self.css_width, "height": self.css_height},
                    "device_scale_factor": self.dpr,
                    "is_mobile": False,
                    "storage_state": auth_info,
                }

                # Add video recording configuration if needed
                if enable_recording:
                    self.temp_video_dir = tempfile.TemporaryDirectory()
                    context_options["record_video_dir"] = self.temp_video_dir.name
                    context_options["record_video_size"] = {
                        "width": self.css_width,
                        "height": self.css_height,
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
        if self.browser_backend == "browserbase":
            # Browserbase's provisioned context doesn't set an explicit viewport
            # (unlike the local new_context() path above), so page.viewport_size
            # is None until set explicitly.
            await self.page.set_viewport_size({"width": self.css_width, "height": self.css_height})
        await self.page.bring_to_front()

        # Register the page listener last -- this listener is used to automatically
        # switch Page handles. Only (re-)register on a genuinely new context: it's
        # attached to self.context itself (via context.on("page", ...)), so on the
        # reused Browserbase context re-registering would stack duplicate listeners
        # on every reset(). setup_dialog_interceptor(), by contrast, is scoped to
        # the current self.page object (captured by value at call time) rather than
        # to the context, so it must be (re-)attached every time regardless --
        # pages opened directly via context.new_page() above (as opposed to
        # browser-triggered popups/new tabs, which setup_global_page_listener's own
        # _handle_new_page callback already re-attaches this to) never fire the
        # context's "page" event and so would otherwise be left without one.
        if is_new_context:
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

        if self.bb and self.bb_session:
            try:
                self.bb.sessions.update(self.bb_session.id, status="REQUEST_RELEASE")
            except Exception as e:
                logger.error(f"Error releasing Browserbase session: {str(e)}")
            self.bb = None
            self.bb_session = None

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
        if not _HANG_GUARD:
            return await self.find_all_clickable_elements()
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
            # env_step_error aborts, and 100% of the 2026-09-05 smoke aborts. With the hang
            # guard on, fall back to a raw CDP capture, which does not wait for fonts.
            if not _HANG_GUARD:
                raise
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
        # change-map (inert unless SLIME_BROWSER_CHANGE_MAP=1): remember the RAW frame, serve the overlaid one
        screenshot_bytes, self._cm_prev = _maybe_change_map(screenshot_bytes, getattr(self, "_cm_prev", None))
        return _maybe_upscale_screenshot(screenshot_bytes)

    def get_all_tabs(self) -> List:
        """
        Get all opened tabs (pages) in the current browser context.

        Returns:
            List of page objects
        """
        if not self.context:
            return []
        return self.context.pages

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

    def get_active_tab(self):
        """
        Get the currently active tab (page).

        Returns:
            The active page object, or None if no page is active
        """
        return self.page

    async def get_active_tab_url(self) -> Optional[str]:
        """
        Get the URL of the currently active tab.

        Returns:
            URL string of the active tab, or None if no page is active
        """
        if not self.page:
            return None
        return self.page.url

    async def get_element_type(self, parameters: Dict[str, Any]) -> str:
        """
        Get the type of element at specified coordinates.

        Args:
            parameters: Dictionary containing x, y coordinates

        Returns:
            str: Element type (tag name)
        """
        if "x" in parameters and "y" in parameters:
            return await self.page.evaluate(
                """
                ({x, y}) => {
                    const element = document.elementFromPoint(x, y);
                    if (!element) return 'none';

                    // Check if select element
                    if (element.tagName.toLowerCase() === 'select') {
                        return 'select';
                    }

                    // Check if option element
                    if (element.tagName.toLowerCase() === 'option') {
                        return 'option';
                    }

                    return element.tagName.toLowerCase();
                }
                """,
                {"x": parameters["x"], "y": parameters["y"]},
            )
        else:
            return "unknown"

    async def _get_element_description(self, x: float, y: float) -> str:
        """Get a short human-readable description of the element at viewport coordinates."""
        try:
            info = await self.page.evaluate(
                """
                ({x, y}) => {
                    const el = document.elementFromPoint(x, y);
                    if (!el) return null;
                    const tag = el.tagName.toLowerCase();
                    const role = el.getAttribute('role') || '';
                    const type = el.getAttribute('type') || '';
                    const ariaLabel = el.getAttribute('aria-label') || '';
                    const checked = typeof el.checked === 'boolean' ? el.checked : null;
                    let text = (el.textContent || '').trim();
                    if (text.length > 30) text = text.substring(0, 30) + '...';
                    return { tag, role, type, ariaLabel, checked, text };
                }
                """,
                {"x": x, "y": y},
            )
            if not info:
                return ""
            desc = f"<{info['tag']}>"
            if info.get("type"):
                desc += f" type={info['type']}"
            if info.get("role"):
                desc += f" role={info['role']}"
            if info.get("checked") is not None:
                desc += " checked" if info["checked"] else " unchecked"
            if info.get("ariaLabel"):
                desc += f" \"{info['ariaLabel']}\""
            elif info.get("text"):
                desc += f" \"{info['text']}\""
            return desc
        except Exception:
            return ""

    async def _get_focused_element_description(self) -> str:
        """Get a short human-readable description of the currently focused DOM element."""
        try:
            info = await self.page.evaluate(
                """() => {
                    const el = document.activeElement;
                    if (!el || el === document.body) return null;
                    const tag = el.tagName.toLowerCase();
                    const role = el.getAttribute('role') || '';
                    const type = el.getAttribute('type') || '';
                    const ariaLabel = el.getAttribute('aria-label') || '';
                    const placeholder = el.getAttribute('placeholder') || '';
                    let text = (el.textContent || '').trim();
                    if (text.length > 30) text = text.substring(0, 30) + '...';
                    return { tag, role, type, ariaLabel, placeholder, text };
                }"""
            )
            if not info:
                return ""
            desc = f"<{info['tag']}>"
            if info.get("type"):
                desc += f" type={info['type']}"
            if info.get("role"):
                desc += f" role={info['role']}"
            if info.get("ariaLabel"):
                desc += f" \"{info['ariaLabel']}\""
            elif info.get("placeholder"):
                desc += f" \"{info['placeholder']}\""
            elif info.get("text"):
                desc += f" \"{info['text']}\""
            return desc
        except Exception:
            return ""

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
        if not _HANG_GUARD or page is None or page.is_closed() or page.url not in _DEAD_POPUP_URLS:
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
            if _HANG_GUARD:
                try:
                    await asyncio.wait_for(
                        self.page.wait_for_load_state("domcontentloaded"), timeout=_HANG_GUARD_A11Y_SECS
                    )
                except Exception as exc:
                    logger.warning("hang guard: new page load wait failed (%s)", type(exc).__name__)
                if await self._recover_from_dead_popup():
                    return
            else:
                await self.page.wait_for_load_state("domcontentloaded")
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
            # Convert absolute coordinates -> CSS Viewport coordinates for point_2d format
            if "point_2d" in parameters and parameters["point_2d"] is not None:
                point = parameters["point_2d"]
                if isinstance(point, list) and len(point) >= 2:
                    # Scale by DPR
                    point[0] = point[0] / self.dpr  # x
                    point[1] = point[1] / self.dpr  # y
                    # Apply smart resize if configured
                    if self.smart_resize_height and self.smart_resize_width:
                        point[0] = point[0] / self.smart_resize_width * self.screen_size[0]
                        point[1] = point[1] / self.smart_resize_height * self.screen_size[1]
                    parameters["point_2d"] = point

            # Also scale start_point_2d and end_point_2d (used by drag)
            for coord_key in ["start_point_2d", "end_point_2d"]:
                if coord_key in parameters and parameters[coord_key] is not None:
                    point = parameters[coord_key]
                    if isinstance(point, list) and len(point) >= 2:
                        point[0] = point[0] / self.dpr
                        point[1] = point[1] / self.dpr
                        if self.smart_resize_height and self.smart_resize_width:
                            point[0] = point[0] / self.smart_resize_width * self.screen_size[0]
                            point[1] = point[1] / self.smart_resize_height * self.screen_size[1]
                        parameters[coord_key] = point

            # Action type mapping table
            action_handlers = {
                "click": self._execute_click,
                "write": self._execute_write,
                "scroll": self._execute_scroll,
                "press_keys": self._execute_press_keys,
                "wait": self._execute_wait,
                "done": self._execute_done,
                "goto_url": self._execute_goto_url,
                "go_back": self._execute_go_back,
                "hover": self._execute_hover,
                "drag": self._execute_drag,
                "new_tab": self._execute_new_tab,
                "switch_tab": self._execute_switch_tab,
                "close_tab": self._execute_close_tab,
                "call_user": self._execute_call_user,
                "fill_and_submit": self._execute_fill_and_submit,
                "open_tab_at_url": self._execute_open_tab_at_url,
                "scroll_repeat": self._execute_scroll_repeat,
                "click_repeat": self._execute_click_repeat,
                "click_and_type": self._execute_click_and_type,
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

    async def _execute_click(self, parameters: Dict[str, Any]) -> tuple[bool, str]:
        """
        Click at specified position.

        Args:
            parameters: Dictionary containing click parameters:
                - point_2d (list[number]): [x, y] coordinates (required)
                - clicks (int): Number of clicks, default 1
                - button (str): "left", "right", or "middle", default "left"

        Returns:
            tuple[bool, str]: (success flag, result message)
        """
        try:
            point_2d = parameters.get("point_2d")
            if not point_2d or len(point_2d) != 2:
                return False, "Failed: `click` requires a valid `point_2d` [x, y]."

            x, y = point_2d[0], point_2d[1]
            clicks = parameters.get("clicks", 1)
            button = parameters.get("button", "left")

            elem_desc = await self._get_element_description(x, y)
            pre_url = self.page.url
            pre_tab_count = len(self.context.pages) if self.context else 0

            await self.page.mouse.click(x, y, button=button, click_count=clicks)
            try:
                await self.page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass  # Page may never reach networkidle (e.g., long-polling)
            await self.page.wait_for_timeout(1000)

            if self._molmoweb_feedback and self._molmoweb_extra is not None:
                # MolmoWeb vocabulary: name the element whose box owns this
                # point, and stop there -- the corpus has no navigation note.
                mw_target = molmoweb_feedback.resolve_click_target(
                    self._molmoweb_axtree, self._molmoweb_extra, x, y
                )
                return True, "Succeed: " + molmoweb_feedback.describe_click(
                    mw_target, button=button, double=(clicks or 1) >= 2
                )

            target = f" on {elem_desc}" if elem_desc else ""
            feedback = f"Succeed: `click`{target} at ({x:.0f}, {y:.0f}) executed."
            post_url = self.page.url
            post_tab_count = len(self.context.pages) if self.context else 0
            if post_url != pre_url:
                feedback += f" Page navigated to {post_url}."
            if post_tab_count > pre_tab_count:
                feedback += f" New tab opened (tab {post_tab_count - 1})."
            if post_url == pre_url and post_tab_count == pre_tab_count:
                feedback += " Note: no visible navigation or new tab detected."
            return True, feedback
            
        except Exception as e:
            return False, f"Failed: `click` execution failed: {e}"

    async def _execute_write(self, parameters: Dict[str, Any]) -> tuple[bool, str]:
        """
        Simulate keyboard typing text.

        Args:
            parameters: Dictionary containing:
                - message (str): Text to type (required)

        Returns:
            tuple[bool, str]: (success flag, result message)
        """
        try:
            message = parameters.get("message", "")
            if not message:
                return False, "Failed: `write` requires a non-empty `message`."

            elem_desc = await self._get_focused_element_description()
            pre_url = self.page.url

            # Clear existing content first
            await self.page.keyboard.press("Control+A")
            await self.page.keyboard.press("Backspace")

            await self.page.keyboard.type(message)
            try:
                await self.page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass
            await self.page.wait_for_timeout(1000)

            if self._molmoweb_feedback:
                return True, "Succeed: " + molmoweb_feedback.describe_type(message)

            target = f" into {elem_desc}" if elem_desc else ""
            feedback = f"Succeed: `write` typed \"{message}\"{target}."
            post_url = self.page.url
            if post_url != pre_url:
                feedback += f" Page navigated to {post_url}."

            try:
                actual_value = await self.page.evaluate(
                    "() => document.activeElement ? (document.activeElement.value || document.activeElement.textContent || '') : ''"
                )
                if actual_value and actual_value != message:
                    displayed_value = truncate_feedback_text(
                        actual_value,
                        DEFAULT_BROWSER_ACTUAL_VALUE_MAX_CHARS,
                    )
                    feedback += (
                        f" Note: the field's actual value is \"{displayed_value}\", "
                        "which differs from the typed text."
                    )
            except Exception:
                pass

            return True, feedback
        except Exception as e:
            return False, f"Failed: `write` execution failed: {e}"

    async def _execute_scroll(self, parameters: Dict[str, Any]) -> tuple[bool, str]:
        """
        Execute scroll operation on the page or a specific element.

        Args:
            parameters: Dictionary containing:
                - direction (str): "up", "down", "left", or "right", default "down"
                - amount (float): Viewport fraction 0.0–1.0, default 0.5
                - point_2d (list[number], optional): [x, y] coordinates of the element to scroll within

        Returns:
            tuple[bool, str]: (success flag, result message)
        """
        try:
            direction = parameters.get("direction", "down")
            amount = parameters.get("amount", 0.5)
            amount = max(0.0, min(1.0, float(amount)))
            point_2d = parameters.get("point_2d")

            if direction in ["up", "down"]:
                delta_x = 0
                delta_y = (
                    -self.css_height * amount
                    if direction == "up"
                    else self.css_height * amount
                )
            else:  # direction in ["left", "right"]
                delta_x = (
                    -self.css_width * amount
                    if direction == "left"
                    else self.css_width * amount
                )
                delta_y = 0

            pre_scroll = await self.page.evaluate("() => ({x: window.scrollX, y: window.scrollY})")

            if point_2d and len(point_2d) == 2:
                x, y = float(point_2d[0]), float(point_2d[1])
                await self.page.mouse.move(x, y)
                await self.page.evaluate(
                    """
                    ({x, y, deltaX, deltaY}) => {
                        let el = document.elementFromPoint(x, y);
                        while (el && el !== document.body && el !== document.documentElement) {
                            const style = window.getComputedStyle(el);
                            const overflowY = style.overflowY;
                            const overflowX = style.overflowX;
                            const isScrollableY =
                                (overflowY === 'auto' || overflowY === 'scroll') &&
                                el.scrollHeight > el.clientHeight;
                            const isScrollableX =
                                (overflowX === 'auto' || overflowX === 'scroll') &&
                                el.scrollWidth > el.clientWidth;
                            if ((deltaY !== 0 && isScrollableY) || (deltaX !== 0 && isScrollableX)) {
                                el.scrollBy(deltaX, deltaY);
                                return;
                            }
                            el = el.parentElement;
                        }
                        window.scrollBy(deltaX, deltaY);
                    }
                    """,
                    {"x": x, "y": y, "deltaX": delta_x, "deltaY": delta_y},
                )
            else:
                await self.page.evaluate(f"window.scrollBy({delta_x}, {delta_y});")

            try:
                await self.page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass
            await self.page.wait_for_timeout(1000)
            target = f" at ({point_2d[0]}, {point_2d[1]})" if point_2d else ""
            feedback = f"Succeed: `scroll` {direction} by {amount:.0%}{target} executed."
            post_scroll = await self.page.evaluate("() => ({x: window.scrollX, y: window.scrollY})")
            dx_actual = post_scroll["x"] - pre_scroll["x"]
            dy_actual = post_scroll["y"] - pre_scroll["y"]
            if self._molmoweb_feedback:
                # Report the pixels actually moved, which is the unit the
                # MolmoWeb corpus uses (this harness natively reports a percent).
                return True, "Succeed: " + molmoweb_feedback.describe_scroll(
                    dx_actual, dy_actual
                )
            if abs(dx_actual) < 1 and abs(dy_actual) < 1 and not point_2d:
                feedback += " Note: page scroll position did not change and may be at a boundary."
            return True, feedback
        except Exception as e:
            return False, f"Failed: `scroll` execution failed: {e}"

    async def _execute_press_keys(self, parameters: Dict[str, Any]) -> tuple[bool, str]:
        """
        Handles all keyboard interactions: single keys, sequences, and hotkeys.

        Args:
            parameters: Dictionary containing:
                - keys (list[str]): Keys to press (required)
                - is_hotkey (bool): If True, press simultaneously; default False
                - presses (int): Repeat count; default 1

        Returns:
            tuple[bool, str]: (success flag, result message)
        """
        try:
            keys = parameters.get("keys", [])
            presses = parameters.get("presses", 1)
            is_hotkey = parameters.get("is_hotkey", False)

            # Normalize input to ensure we always work with a list
            if isinstance(keys, str):
                keys = [keys]
            if not keys:
                return False, "Failed: `press_keys` requires a non-empty `keys` list."

            # Validate and clean keys
            legal_keys = []
            for k in keys:
                valid_key = validate_and_correct_key(k)
                if valid_key:
                    legal_keys.append(valid_key)

            if not legal_keys:
                return False, f"Failed: no valid keys found in {keys}."

            pre_url = self.page.url

            # Execution Logic
            if is_hotkey:
                # Simultaneous: Join keys (e.g., ["Control", "c"] -> "Control+c")
                key_combo = "+".join(legal_keys)
                for _ in range(presses):
                    await self.page.keyboard.press(key_combo)
            else:
                # Sequential: Press one by one (e.g., ["a", "b"] -> press "a", then "b")
                for key in legal_keys:
                    for _ in range(presses):
                        await self.page.keyboard.press(key)
                        # Small delay between sequential key presses often helps realism
                        if len(legal_keys) > 1:
                            await self.page.wait_for_timeout(100)

            try:
                await self.page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass
            await self.page.wait_for_timeout(1000)
            if self._molmoweb_feedback:
                return True, "Succeed: " + molmoweb_feedback.describe_press(legal_keys)

            feedback = f"Succeed: `press_keys` {legal_keys} executed."
            post_url = self.page.url
            if post_url != pre_url:
                feedback += f" Page navigated to {post_url}."
            return True, feedback
        except Exception as e:
            return False, f"Failed: `press_keys` execution failed: {e}"

    async def _execute_wait(self, parameters: Dict[str, Any]) -> tuple[bool, str]:
        """
        Wait for specified number of seconds.

        Args:
            parameters: Dictionary containing:
                - seconds (int): Seconds to wait, default 3

        Returns:
            tuple[bool, str]: (success flag, result message)
        """
        try:
            seconds = parameters.get("seconds", 3)
            if _HANG_GUARD and int(seconds) > _HANG_GUARD_WAIT_MAX_SECS:
                seconds = _HANG_GUARD_WAIT_MAX_SECS
            await self.page.wait_for_timeout(int(seconds) * 1000)
            if self._molmoweb_feedback:
                return True, "Succeed: " + molmoweb_feedback.describe_wait()
            return True, f"Succeed: waited {seconds} seconds."
        except Exception as e:
            return False, f"Failed: `wait` execution failed: {e}"

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
        if self._molmoweb_feedback:
            return True, "Succeed: " + molmoweb_feedback.describe_done(response)
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
            if self._molmoweb_feedback:
                return True, "Succeed: " + molmoweb_feedback.describe_go_back()
            return True, "Succeed: `go_back` executed."
        except Exception as e:
            return False, f"Failed: `go_back` execution failed: {e}"

    async def _execute_hover(self, parameters: Dict[str, Any]) -> tuple[bool, str]:
        """
        Move mouse to specified position.

        Args:
            parameters: Dictionary containing:
                - point_2d (list[number]): [x, y] coordinates (required)

        Returns:
            tuple[bool, str]: (success flag, result message)
        """
        try:
            point_2d = parameters.get("point_2d")
            if not point_2d or len(point_2d) != 2:
                return False, "Failed: `hover` requires a valid `point_2d` [x, y]."

            x, y = float(point_2d[0]), float(point_2d[1])
            elem_desc = await self._get_element_description(x, y)
            await self.page.mouse.move(x, y)
            await self.page.wait_for_timeout(2000)
            target = f" on {elem_desc}" if elem_desc else ""
            return True, f"Succeed: `hover`{target} at ({x:.0f}, {y:.0f}) executed."
        except Exception as e:
            return False, f"Failed: `hover` execution failed: {e}"

    async def _execute_drag(self, parameters: Dict[str, Any]) -> tuple[bool, str]:
        """
        Execute a drag-and-drop operation using coordinates.

        Args:
            parameters: Dictionary containing:
                - start_point_2d (list[number]): [x, y] start coordinates (required)
                - end_point_2d (list[number]): [x, y] end coordinates (required)
                - steps (int): Intermediate mouse steps for smoothness, default 10

        Returns:
            tuple[bool, str]: (success flag, result message)
        """
        try:
            start_point = parameters.get("start_point_2d")
            end_point = parameters.get("end_point_2d")
            steps = parameters.get("steps", 10)

            if not start_point or len(start_point) != 2:
                return False, "Failed: `drag` requires a valid `start_point_2d` [x, y]."
            if not end_point or len(end_point) != 2:
                return False, "Failed: `drag` requires a valid `end_point_2d` [x, y]."

            from_x, from_y = float(start_point[0]), float(start_point[1])
            to_x, to_y = float(end_point[0]), float(end_point[1])

            await self.page.mouse.move(x=from_x, y=from_y)
            await self.page.mouse.down(button="left")
            await self.page.mouse.move(x=to_x, y=to_y, steps=steps)
            await self.page.mouse.up(button="left")
            await self.page.wait_for_timeout(500)
            return True, "Succeed: `drag` executed."
        except Exception as e:
            return False, f"Failed: `drag` execution failed: {e}"

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
            if self._molmoweb_feedback:
                return True, "Succeed: " + molmoweb_feedback.describe_goto(url)
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

    async def _execute_call_user(self, parameters: Dict[str, Any]) -> tuple[bool, str]:
        """
        Pauses execution to ask the user a question or request confirmation.

        Args:
            parameters: Dictionary containing:
                - question (str): The question to ask the user (required)

        Returns:
            tuple[bool, str]: (success flag, result message with user's response)
        """
        question = parameters.get("question")

        if not question:
            return False, "Failed: no question provided. You must ask or inform the user something."

        print(f"\n\033[93m[AGENT ASKING USER]\033[0m {question}")

        user_response = input("Your Answer: ").strip()

        if not user_response:
            return False, "User provided no input. Please try asking again or decide on a default."

        print(f"\033[92m[USER ANSWERED]\033[0m {user_response}\n")
        return True, f"User response: {user_response}"

    async def _execute_fill_and_submit(self, parameters: Dict[str, Any]) -> tuple[bool, str]:
        """
        Compound action for the action-space ablation: click an input field, type
        into it, then press Enter. Equivalent to click + write + press_keys(["Enter"]).

        Args:
            parameters: Dictionary containing:
                - point_2d (list[number]): [x, y] coordinates of the input field (required)
                - message (str): Text to type (required)

        Returns:
            tuple[bool, str]: (success flag, result message)
        """
        click_ok, click_msg = await self._execute_click(parameters)
        if not click_ok:
            return False, f"Failed: `fill_and_submit` click step failed: {click_msg}"

        write_ok, write_msg = await self._execute_write(parameters)
        if not write_ok:
            return False, f"Failed: `fill_and_submit` write step failed: {write_msg}"

        press_ok, press_msg = await self._execute_press_keys({"keys": ["Enter"]})
        if not press_ok:
            return False, f"Failed: `fill_and_submit` press_keys step failed: {press_msg}"

        return True, f"Succeed: `fill_and_submit` executed ({click_msg} {write_msg} {press_msg})"

    async def _execute_open_tab_at_url(self, parameters: Dict[str, Any]) -> tuple[bool, str]:
        """
        Compound action for the action-space ablation: open a new tab and navigate
        it to a URL. Equivalent to new_tab + switch_tab + goto_url (new_tab already
        switches to the new tab, so no separate switch_tab call is needed).

        Args:
            parameters: Dictionary containing:
                - url (str): The URL to navigate to in the new tab (required)

        Returns:
            tuple[bool, str]: (success flag, result message)
        """
        new_tab_ok, new_tab_msg = await self._execute_new_tab({})
        if not new_tab_ok:
            return False, f"Failed: `open_tab_at_url` new_tab step failed: {new_tab_msg}"

        goto_ok, goto_msg = await self._execute_goto_url(parameters)
        if not goto_ok:
            return False, f"Failed: `open_tab_at_url` goto_url step failed: {goto_msg}"

        return True, f"Succeed: `open_tab_at_url` executed ({new_tab_msg} {goto_msg})"

    async def _execute_scroll_repeat(self, parameters: Dict[str, Any]) -> tuple[bool, str]:
        """
        Compound action for the MolmoWeb repetition-atomization experiment:
        repeat a full-viewport scroll in the same direction `times` times.
        Equivalent to calling scroll(direction, amount=1.0) `times` times in a
        row. Implemented as a loop over the existing single-scroll handler
        (each internal call keeps its own unclamped 1.0 amount) rather than
        passing a summed `amount` to `_execute_scroll`, since that method
        clamps `amount` to [0.0, 1.0] per call and would silently truncate a
        larger cumulative value to a single viewport.

        Args:
            parameters: Dictionary containing:
                - direction (str): "up", "down", "left", or "right" (required)
                - times (int): Number of full-viewport scrolls to perform (required, 2-5)

        Returns:
            tuple[bool, str]: (success flag, result message)
        """
        direction = parameters.get("direction")
        times = parameters.get("times")
        if direction not in ("up", "down", "left", "right"):
            return False, "Failed: `scroll_repeat` requires `direction` in up/down/left/right."
        try:
            times = int(times)
        except (TypeError, ValueError):
            return False, "Failed: `scroll_repeat` requires an integer `times` between 2 and 5."
        if not (2 <= times <= 5):
            return False, "Failed: `scroll_repeat` requires `times` between 2 and 5 (use `scroll` for a single scroll)."

        messages = []
        for i in range(times):
            ok, msg = await self._execute_scroll({"direction": direction, "amount": 1.0})
            if not ok:
                return False, f"Failed: `scroll_repeat` step {i + 1}/{times} failed: {msg}"
            messages.append(msg)

        if self._molmoweb_feedback:
            return True, "Succeed: " + molmoweb_feedback.describe_scroll_repeat(direction, times)
        return True, f"Succeed: `scroll_repeat` {direction} x{times} executed ({' '.join(messages)})"

    async def _execute_click_repeat(self, parameters: Dict[str, Any]) -> tuple[bool, str]:
        """
        Compound action for the MolmoWeb repetition-atomization experiment:
        repeat a single left click at the same point `times` times. Equivalent
        to calling click(point_2d, clicks=1) `times` times in a row.
        Implemented as a loop over the existing single-click handler (each
        internal call keeps `clicks=1`) rather than passing `clicks=times` to
        `_execute_click`, since that method forwards `clicks` straight to
        Playwright's `mouse.click(click_count=...)`, which fires genuine
        multi-click DOM semantics (e.g. `click_count=2` triggers a real
        `dblclick` event) rather than N independent single-click events.

        Args:
            parameters: Dictionary containing:
                - point_2d (list[number]): [x, y] coordinates (required)
                - times (int): Number of single clicks to perform (required, 2-5)

        Returns:
            tuple[bool, str]: (success flag, result message)
        """
        point_2d = parameters.get("point_2d")
        times = parameters.get("times")
        if not point_2d or len(point_2d) != 2:
            return False, "Failed: `click_repeat` requires a valid `point_2d` [x, y]."
        try:
            times = int(times)
        except (TypeError, ValueError):
            return False, "Failed: `click_repeat` requires an integer `times` between 2 and 5."
        if not (2 <= times <= 5):
            return False, "Failed: `click_repeat` requires `times` between 2 and 5 (use `click` for a single click)."

        messages = []
        for i in range(times):
            ok, msg = await self._execute_click({"point_2d": point_2d, "clicks": 1, "button": "left"})
            if not ok:
                return False, f"Failed: `click_repeat` step {i + 1}/{times} failed: {msg}"
            messages.append(msg)

        if self._molmoweb_feedback:
            return True, "Succeed: " + molmoweb_feedback.describe_click_repeat(times)
        return True, f"Succeed: `click_repeat` x{times} executed ({' '.join(messages)})"

    async def _execute_click_and_type(self, parameters: Dict[str, Any]) -> tuple[bool, str]:
        """
        Compound action for the MolmoWeb repetition-atomization experiment:
        click an input field, then type into it (no submit). Equivalent to
        click + write, without the press_keys(["Enter"]) step that
        `fill_and_submit` adds -- use this when the field isn't submitted by
        pressing Enter (e.g. a form with a separate submit button).

        Args:
            parameters: Dictionary containing:
                - point_2d (list[number]): [x, y] coordinates of the input field (required)
                - message (str): Text to type (required)

        Returns:
            tuple[bool, str]: (success flag, result message)
        """
        click_ok, click_msg = await self._execute_click(parameters)
        if not click_ok:
            return False, f"Failed: `click_and_type` click step failed: {click_msg}"

        write_ok, write_msg = await self._execute_write(parameters)
        if not write_ok:
            return False, f"Failed: `click_and_type` write step failed: {write_msg}"

        if self._molmoweb_feedback:
            return True, "Succeed: " + molmoweb_feedback.describe_click_and_type()
        return True, f"Succeed: `click_and_type` executed ({click_msg} {write_msg})"

    async def step(self, action_list: list[dict]):
        """
        Execute environment step with given actions (async version).
        
        This overrides the base class synchronous step() method.
        
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

            # Snapshot the accessibility tree + element boxes before the actions
            # run: a click must be described by the element that was there when
            # the model chose the coordinate, not by whatever the click produced.
            # Extraction failures are non-fatal -- descriptions fall back to the
            # harness's native strings rather than breaking the step.
            if self._molmoweb_feedback:
                try:
                    _extract = molmoweb_feedback.extract_axtree(self.page)
                    if _HANG_GUARD:
                        _extract = asyncio.wait_for(_extract, timeout=_HANG_GUARD_A11Y_SECS)
                    self._molmoweb_axtree, self._molmoweb_extra = await _extract
                except Exception as exc:
                    self._molmoweb_axtree = self._molmoweb_extra = None
                    logger.warning(
                        f"MolmoWeb feedback: axtree extraction failed "
                        f"({type(exc).__name__}); using native feedback this step."
                    )

            for action in action_list:
                if _HANG_GUARD:
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
                else:
                    flag, msg = await self.execute_single_action(action)
                info["tool_responses"].append({
                    "tool_name": action["name"],
                    "tool_response": msg
                })
                terminated = True if (action["name"] == "done" and flag) else False
            
            if _HANG_GUARD:
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
