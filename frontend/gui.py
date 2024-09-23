import asyncio
import time
import streamlit as st
import models, api

api_llm = api.API_LLM()

# Initialize chat history
if "messages" not in st.session_state:
    # models.Assistant_Message
    st.session_state.messages = []

# Initalize chat disable
if "is_chat_input_disabled" not in st.session_state:
    st.session_state.is_chat_input_disabled = False


# Streamed response emulator
def response_generator(content: models.Assistant_Message):
    message_content = content.response["message"]
    ref = content.references
    i = 1

    if len(ref) > 0:
        message_content += "  \n  Tham khảo tại:  \n"
        for ref_content in ref:
            message_content += f"[{i}]. {ref_content['title']} ({ref_content['url']})  \n"
            i += 1

    for word in message_content.split(" "):
        yield word + " "
        time.sleep(0.05)


# Send message
async def send_message(message: models.Message) -> models.Assistant_Message:
    response = await api_llm.make_request("send_message", message)
    assit_message = models.Assistant_Message(**response)
    return assit_message


async def main():
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Disable prompt while assistant responsing
    if len(st.session_state.messages) == 0:
        st.session_state.is_chat_input_disabled = False
    else:
        if st.session_state.messages[-1]["role"] == "user":
            st.session_state.is_chat_input_disabled = True
        else:
            st.session_state.is_chat_input_disabled = False

    # Accept user input
    if prompt := st.chat_input(
        "What is up?", disabled=st.session_state.is_chat_input_disabled
    ):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            message = models.Message(
                message=prompt, history_count=len(st.session_state.messages)
            )

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                assist_response = await send_message(message)

            full_response = st.write_stream(response_generator(assist_response))
            # Add assistant response to chat history
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )


if __name__ == "__main__":
    asyncio.run(main())
