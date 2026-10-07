"""Execute the Responses computer action contract without lossy tool conversion."""
import asyncio

KEYS = {'CTRL': 'Control', 'CONTROL': 'Control', 'CMD': 'Meta', 'COMMAND': 'Meta',
        'META': 'Meta', 'ALT': 'Alt', 'SHIFT': 'Shift', 'ENTER': 'Enter',
        'RETURN': 'Enter', 'ESC': 'Escape', 'ESCAPE': 'Escape', 'SPACE': 'Space',
        'BACKSPACE': 'Backspace', 'DELETE': 'Delete', 'TAB': 'Tab',
        'UP': 'ArrowUp', 'DOWN': 'ArrowDown', 'LEFT': 'ArrowLeft', 'RIGHT': 'ArrowRight',
        'HOME': 'Home', 'END': 'End', 'PAGEUP': 'PageUp', 'PAGEDOWN': 'PageDown'}


def key(value):
    return KEYS.get(value.upper(), value)


async def execute(page, action):
    kind = action['type']
    if kind in ('click', 'double_click'):
        modifiers = [key(k) for k in (action.get('keys') or [])]
        try:
            for k in modifiers:
                await page.keyboard.down(k)
            await page.mouse.click(action['x'], action['y'],
                                   button=action.get('button') or 'left',
                                   click_count=2 if kind == 'double_click' else 1)
        finally:
            for k in reversed(modifiers):
                await page.keyboard.up(k)
    elif kind == 'type':
        await page.keyboard.insert_text(action['text'])
    elif kind == 'keypress':
        await page.keyboard.press('+'.join(key(k) for k in action['keys']))
    elif kind == 'move':
        await page.mouse.move(action['x'], action['y'])
    elif kind == 'scroll':
        await page.mouse.move(action['x'], action['y'])
        await page.mouse.wheel(action.get('scroll_x') or 0, action.get('scroll_y') or 0)
    elif kind == 'drag':
        path = action['path']
        if len(path) < 2:
            raise ValueError('Drag requires at least two points')
        await page.mouse.move(path[0]['x'], path[0]['y'])
        await page.mouse.down()
        try:
            for point in path[1:]:
                await page.mouse.move(point['x'], point['y'])
        finally:
            await page.mouse.up()
    elif kind == 'wait':
        await asyncio.sleep(3)
    elif kind != 'screenshot':
        raise ValueError(f'Unsupported native computer action: {kind}')


def install():
    from openwebrl.env.web_env import WebEnv
    original = WebEnv.execute_single_action

    async def dispatch(self, action):
        if action['name'] != 'computer':
            return await original(self, action)
        assert self.dpr == 1 and self.screen_size == (1280, 1000)
        try:
            await execute(self.page, action['args'])
            return True, 'Succeed: native ' + action['args']['type']
        except Exception as exc:
            return False, f'Failed: native {action["args"]["type"]}: {type(exc).__name__}: {exc}'

    WebEnv.execute_single_action = dispatch
