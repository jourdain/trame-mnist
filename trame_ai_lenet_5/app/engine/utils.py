import asyncio
from trame import state
from trame.internal.app import get_app_instance


async def monitor_state_queue(queue):
    _app = get_app_instance()
    _process_running = True
    while _process_running:
        if queue.empty():
            await asyncio.sleep(1)
        else:
            msg = queue.get_nowait()
            if isinstance(msg, str):
                # command
                if msg == "stop":
                    _process_running = False
            else:
                # state update (dict)
                state.update(msg)
                state.flush(*list(msg.keys()))

                for key in msg.keys(): # Hack
                    callbacks = _app._change_callbacks.get(key, [])
                    for fn in callbacks:
                        fn(**_app.state)
