import asyncio
from .ml import prediction_reload
from trame import state


async def queue_to_state(queue, *tasks):
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
                # Need to monitor as we are outside of client/server update
                with state.monitor():
                    # state update (dict)
                    state.update(msg)

    await asyncio.gather(*tasks)

    # Make sure we can go to prediction
    state.prediction_available = prediction_reload()
    state.testing_count = 0
    state.flush("prediction_available", "testing_count")
