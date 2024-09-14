import streamlit as st
import api
import models

api_llm = api.API_LLM()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Streamed response emulator
async def send_message(message: models.Message):
    return api_llm.make_request("send_message", message)

async def main():
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
            # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message = models.Message(message=prompt, history_count=len(st.session_state.messages))
            response = st.write_stream(send_message(message))
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()