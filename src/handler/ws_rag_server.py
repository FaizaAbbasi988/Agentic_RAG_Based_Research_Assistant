# src/handler/ws_rag_server.py

import asyncio
import json
import websockets

from src.model_service.model_service import ResearchAssistantRAG

research_asssistant = ResearchAssistantRAG()

HEARTBEAT_INTERVAL = 5

async def client_handler(websocket):
    client_id = id(websocket)
    loop = asyncio.get_running_loop()

    socket_lock = asyncio.Lock()

    async def safe_send(payload: dict):
        "Thread-safe helper to send JSON data"
        async with socket_lock:
            try:
                await websocket.send(json.dumps(payload))
            except websockets.exceptions.ConnectionClosed:
                pass
            except Exception as e:
                print(f"Send Error: {e}")

    async def send_heartbeat():
        while True:
            try:
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                await safe_send({"type":"PING"})
            except asyncio.CancelledError:
                break
            except Exception:
                break
    
    print(f"New connection : {websocket.remote_address}")
    heartbeat_task = asyncio.create_task(send_heartbeat())

    try:
        async for data in websocket:
            try:
                message = json.loads(data)
                query = message.get("input")
                user_id = message.get("user_id", "default_user")
                print(f"Request from {user_id}: {query}")

                def status_callback(msg_text : str):
                    asyncio.run_coroutine_threadsafe(
                        safe_send({"type": "Update", "content":msg_text}),
                        loop
                    )
                ai_response = await asyncio.to_thread(
                    research_asssistant.invoke,
                    user_id=user_id,
                    query=query,
                    on_update=status_callback
                )
                await safe_send({"type":"Response", "content": ai_response})
                await safe_send({"type": "DONE"})

            except json.JSONDecodeError:
                await safe_send({"type": "ERROR", "content": "Invalid JSON"})
            except Exception as e:
                print(f"⚠️ Processing Error: {e}")
                await safe_send({"type": "ERROR", "content": str(e)})

    except websockets.exceptions.ConnectionClosed:
        print(f"🔌 Connection closed: {websocket.remote_address}")
    except Exception as e:
        print(f"⚠️ Critical Error: {str(e)}")
    finally:
        heartbeat_task.cancel()
        print(f"🛑 Handler finished for {websocket.remote_address}")

async def main():
    # Set the host and port
    host = "0.0.0.0"  # Listen on all interfaces
    port = 8765
    
    print(f"🚀 Research Assistant Server starting on ws://{host}:{port}")
    
    async with websockets.serve(client_handler, host, port):
        await asyncio.Future()  # This keeps the server running forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user.")
