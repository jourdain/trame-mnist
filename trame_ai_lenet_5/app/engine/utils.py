import asyncio
from trame import state
from trame.internal.app import get_app_instance


async def monitor_state_queue(queue, training_task):
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
                with state.monitor():
                    # Need to monitor as we are outside of client/server update
                    state.update(msg)

    await training_task
