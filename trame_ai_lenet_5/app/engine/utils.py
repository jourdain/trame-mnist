import asyncio
from trame import state


async def monitor_state_queue(queue):
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
