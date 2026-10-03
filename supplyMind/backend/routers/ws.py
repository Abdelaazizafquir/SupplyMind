"""WebSocket router — streams simulation progress to the frontend"""
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from tasks.simulation import register_listener, unregister_listener

router = APIRouter()


@router.websocket("/simulation/{sim_id}")
async def simulation_ws(websocket: WebSocket, sim_id: str):
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue()
    register_listener(sim_id, queue)

    try:
        while True:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=30)
                await websocket.send_json(message)
                if message.get("type") in ("completed", "failed"):
                    break
            except asyncio.TimeoutError:
                # Send keep-alive ping
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    finally:
        unregister_listener(sim_id, queue)
        await websocket.close()
