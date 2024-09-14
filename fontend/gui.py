import asyncio
import time
import streamlit as st
import models, api

api_llm = api.API_LLM()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Streamed response emulator
def response_generator(content: str):
    for word in content.split():
        yield word + " "
        time.sleep(0.05)

# Send message
async def send_message(message: models.Message) -> models.Assistant_Message:
    response = await api_llm.make_request("send_message", message)
    print(response)
    assit_message = models.Assistant_Message(**response)
    return assit_message


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
            message = models.Message(
                message=prompt, history_count=len(st.session_state.messages)
            )
            assist_response = await send_message(message)
            st.write_stream(response_generator(assist_response.message))
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": assist_response})


if __name__ == "__main__":
    asyncio.run(main())
