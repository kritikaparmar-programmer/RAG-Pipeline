import streamlit as st
import asyncio
st.session_state.clicked = False

vectorstore_created = False
flag=False

def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return asyncio.get_event_loop()

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

web_path = st.sidebar.text_input("Enter website url")
if web_path:
    rag_pipe = load_rag_pipeline(web_path)
    st.session_state.clicked=True
