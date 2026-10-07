"""Run OpenAI Responses `computer` actions with Playwright in the browser env server (no coordinate conversion).

install() patches WebEnv so actions named "computer" go to execute(); every other tool keeps the harness path.
"""
import asyncio

KEYS = {'CTRL': 'Control', 'CONTROL': 'Control', 'CMD': 'Meta', 'COMMAND': 'Meta',
        'META': 'Meta', 'ALT': 'Alt', 'SHIFT': 'Shift', 'ENTER': 'Enter',
        'RETURN': 'Enter', 'ESC': 'Escape', 'ESCAPE': 'Escape', 'SPACE': 'Space',
        'BACKSPACE': 'Backspace', 'DELETE': 'Delete', 'TAB': 'Tab',
        'UP': 'ArrowUp', 'DOWN': 'ArrowDown', 'LEFT': 'ArrowLeft', 'RIGHT': 'ArrowRight',
        'HOME': 'Home', 'END': 'End', 'PAGEUP': 'PageUp', 'PAGEDOWN': 'PageDown'}
KEYS.update(ARROWUP='ArrowUp', ARROWDOWN='ArrowDown', ARROWLEFT='ArrowLeft',
            ARROWRIGHT='ArrowRight', CAPSLOCK='CapsLock')


def key(value):
    return KEYS.get(value.upper(), value)


async def execute(page, action):
    kind = action['type']
    modifiers = [] if kind == 'keypress' else [key(k) for k in (action.get('keys') or [])]
    held = []
    try:
        for k in modifiers:
            await page.keyboard.down(k)
            held.append(k)
        if kind in ('click', 'double_click'):
            button = action.get('button') or 'left'
            if button in ('back', 'forward'):
                session = await page.context.new_cdp_session(page)
                try:
                    mask = sum({'Alt':1,'Control':2,'Meta':4,'Shift':8}.get(k,0) for k in held)
                    for event, buttons in [('mousePressed',8 if button == 'back' else 16),('mouseReleased',0)]:
                        await session.send('Input.dispatchMouseEvent', {
                            'type':event,'button':button,'buttons':buttons,'x':action['x'],
                            'y':action['y'],'clickCount':1,'modifiers':mask})
                finally:
                    await session.detach()
            else:
                await page.mouse.click(action['x'],action['y'],button='middle' if button == 'wheel' else button,
                                       click_count=2 if kind == 'double_click' else 1)
        elif kind == 'type':
            await page.keyboard.insert_text(action['text'])
        elif kind == 'keypress':
            await page.keyboard.press('+'.join(key(k) for k in action['keys']))
        elif kind == 'move':
            await page.mouse.move(action['x'],action['y'])
        elif kind == 'scroll':
            await page.mouse.move(action['x'],action['y'])
            await page.mouse.wheel(action.get('scroll_x') or 0,action.get('scroll_y') or 0)
        elif kind == 'drag':
            path=action['path']
            if len(path)<2:
                raise ValueError('Drag requires at least two points')
            await page.mouse.move(path[0]['x'],path[0]['y'])
            await page.mouse.down()
            try:
                for point in path[1:]:
                    await page.mouse.move(point['x'],point['y'])
            finally:
                await page.mouse.up()
        elif kind == 'wait':
            await asyncio.sleep(3)
        elif kind != 'screenshot':
            raise ValueError(f'Unsupported native computer action: {kind}')
    finally:
        for k in reversed(held):
            await page.keyboard.up(k)


def install():
    from openwebrl.env.web_env import WebEnv
    original=WebEnv.execute_single_action

    async def dispatch(self,action):
        if action['name']!='computer':
            return await original(self,action)
        assert self.dpr==1 and self.screen_size==(1280,1000)
        try:
            await execute(self.page,action['args'])
            return True,'Succeed: native '+action['args']['type']
        except Exception as exc:
            return False,f'Failed: native {action["args"]["type"]}: {type(exc).__name__}: {exc}'

    WebEnv.execute_single_action=dispatch
