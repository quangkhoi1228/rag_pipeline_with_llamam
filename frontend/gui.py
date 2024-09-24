import asyncio
import time
import streamlit as st
import models, api
from component import st_horizontal

api_llm = api.API_LLM()

# Initialize chat history
if "messages" not in st.session_state:
    # models.Assistant_Message
    st.session_state.messages = []

# Initalize chat disable
if "regenerate" not in st.session_state:
    st.session_state.regenerate = False


# Streamed response emulator
def response_generator(content: models.Assistant_Message):
    message_content = content.response["message"]
    ref = content.references
    i = 1

    if len(ref) > 0:
        message_content += "  \n  Tham khảo tại:  \n"
        for ref_content in ref:
            message_content += (
                f"[{i}]. {ref_content['title']} ({ref_content['url']})  \n"
            )
            i += 1

    for word in message_content.split(" "):
        yield word + " "
        time.sleep(0.05)


# Send message
async def send_message(message: models.Message) -> models.Assistant_Message:
    response = await api_llm.make_request("send_message", message)
    assit_message = models.Assistant_Message(**response)
    return assit_message


# Regenerate response
async def regenerate_response(message: models.Message) -> models.Assistant_Message:
    response = await api_llm.make_request("regenerate_response", message)
    assit_message = models.Assistant_Message(**response)
    return assit_message


# Change status of regenerate
def regenerate():
    st.session_state.regenerate = True
    st.session_state.messages.pop()


# Get last message
def get_message():
    return models.Message(
        message=st.session_state.messages[-1]["content"],
        history_count=len(st.session_state.messages),
    )


async def main():

    # Display chat messages from history on app rerun
    for message_history in st.session_state.messages:
        with st.chat_message(message_history["role"]):
            st.markdown(message_history["content"])

    # When user click regenerate
    if st.session_state.regenerate:
        # Reset variable
        st.session_state.regenerate = False

        with st.chat_message("assistant"):
            message = get_message()
            with st.spinner("Thinking..."):
                assist_response = await regenerate_response(message)

            full_response = st.write_stream(response_generator(assist_response))
            # Add assistant response to chat history
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )

    # Accept user input
    if prompt := st.chat_input("Hãy hỏi gì đó đi"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):

            message = get_message()

            with st.spinner("Thinking..."):
                assist_response = await send_message(message)

            full_response = st.write_stream(response_generator(assist_response))
            # Add assistant response to chat history
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )

    if len(st.session_state.messages) > 0:
        with st_horizontal():
            # Regenerate response of assistant
            st.button(
                ":material/replay:", help="Tạo lại câu trả lời", on_click=regenerate
            )

            # TODO: Like
            st.button(":material/thumb_up_alt:", help="Câu này được đấy")
            # TODO: Dislike
            st.button(":material/thumb_down_alt:", help="Hmm, còn hơi non")


if __name__ == "__main__":
    asyncio.run(main())
