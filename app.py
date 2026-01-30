import streamlit as st
import json
import asyncio
import websockets

# Configuration
WS_URL = "ws://127.0.0.1:8765"

st.set_page_config(page_title="Agentic Research Assistant", page_icon="💼")

# --- Sidebar: User ID Management ---
st.sidebar.title("Configuration")
# User manually enters their ID here
custom_user_id = st.sidebar.text_input("Enter User ID", value="faiza_01") 

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- WebSocket Communication Logic ---
async def communicate_with_backend(query, user_id):
    try:
        async with websockets.connect(WS_URL) as websocket:
            # We send the user-defined ID from the sidebar
            payload = {
                "user_id": user_id, 
                "input": query
            }
            # Send as JSON string
            await websocket.send(json.dumps(payload)) 

            status_placeholder = st.empty()
            full_response = ""

            while True:
                message = await websocket.recv()
                data = json.loads(message)
                
                if data["type"] == "Update":
                    # Displaying continuous updates from agents
                    status_placeholder.info(f"Agent Status: {data['content']}")
                elif data["type"] == "Response":
                    full_response = data["content"]
                elif data["type"] == "DONE":
                    status_placeholder.empty()
                    break
                elif data["type"] == "ERROR":
                    st.error(f"Error: {data['content']}")
                    break
            
            return full_response
    except Exception as e:
        st.error(f"Connection failed: {e}")
        return None

# --- Main Chat UI ---
st.title("💼 RAG Based Research Assistant")
st.caption(f"Active Session: {custom_user_id}")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask about latest research and more..."):
    # Check if ID is provided
    if not custom_user_id:
        st.warning("Please enter a User ID in the sidebar first!")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Pass the custom_user_id to the communication function
        response = asyncio.run(communicate_with_backend(prompt, custom_user_id))
        if response:
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})