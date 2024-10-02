import asyncio
import time
import streamlit as st
import models
import api
from component import st_horizontal

api_llm = api.API_LLM()

# Initialize chat history
if "messages" not in st.session_state:
    # models.Assistant_Message
    st.session_state.messages = []

# Initalize chat disable
if "regenerate" not in st.session_state:
    st.session_state.regenerate = False

# Initalize chat feedback
if "feedback" not in st.session_state:
    st.session_state.feedback = {"is_feedbacked": False, "feedback": None}

# Initalize chat history
if "is_first_time" not in st.session_state:
    st.session_state.is_first_time = True


# Streamed response emulator
def response_generator(content: models.Assistant_Message):
    message_content = content.response["message"]

    # References for RAG
    # ref = content.references
    # i = 1

    # if len(ref) > 0:
    #     message_content += "  \n  Tham khảo tại:  \n"
    #     for ref_content in ref:
    #         message_content += (
    #             f"[{i}]. {ref_content['title']} ({ref_content['url']})  \n"
    #         )
    #         i += 1

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


# Regenerate response
async def feedback(feedback: models.Feedback):
    await api_llm.make_request("feedback", feedback)


# Get messages history
async def messages_history() -> list[models.Assistant_Respone]:
    response_list = await api_llm.make_request("messages_history")
    assit_message = [models.Assistant_Respone(**response) for response in response_list]
    return assit_message


# Clear chat
async def clear_chat():
    await api_llm.make_request("clear_chat")


# Get last message
def get_message():
    return models.Message(
        message=st.session_state.messages[-1]["content"],
        history_count=len(st.session_state.messages),
        faq_id=st.session_state.messages[-1]["faq_id"],
    )


# Check that response have feedback
def is_feedbacked():
    assistant_respone = st.session_state.messages[-1]
    if assistant_respone["feedback"] is not None:
        return True
    else:
        return False


# =============================== Button Function ============================================
# Change status of regenerate
def regenerate():
    st.session_state.regenerate = True
    st.session_state.messages.pop()


# Like button
def like():
    assistant_respone = st.session_state.messages[-1]
    st.session_state.feedback = {
        "is_feedbacked": True,
        "feedback": models.Feedback(
            assistant_respone["faq_id"], assistant_respone["faq_pool_id"], "good"
        ),
    }
    st.session_state.messages[-1] = {**
                                     st.session_state.messages[-1], feedback: "good"}


# Dislike button
def dislike():
    assistant_respone = st.session_state.messages[-1]
    st.session_state.feedback = {
        "is_feedbacked": True,
        "feedback": models.Feedback(
            assistant_respone["faq_id"], assistant_respone["faq_pool_id"], "bad"
        ),
    }
    st.session_state.messages[-1] = {
        **st.session_state.messages[-1],
        feedback: "bad",
    }


async def main():
    if st.session_state.is_first_time:
        with st.spinner("Waitting"):
            history = await messages_history()
        for message in history:
            chat_box = {}
            if message.sender == "user":
                chat_box = {
                    "role": "user",
                    "content": message.message,
                    "faq_id": None,
                }
            else:
                chat_box = {
                    "role": "assistant",
                    "content": message.message,
                    "faq_id": None,
                    "faq_pool_id": None,
                    "feedback": None,
                }
            st.session_state.messages.append(chat_box)
        st.session_state.is_first_time = False

    with st.sidebar:
        if st.button(":material/clear: Xoá hội thoại"):
            st.session_state.messages = []
            await clear_chat()

    # Display chat messages from history on app rerun
    for message_history in st.session_state.messages:
        with st.chat_message(message_history["role"]):
            st.markdown(message_history["content"])

    # When user click regenerate
    if st.session_state.regenerate:
        # Reset variable
        st.session_state.feedback = {"is_feedbacked": False, "feedback": None}
        st.session_state.regenerate = False

        with st.chat_message("assistant"):
            message = get_message()
            with st.spinner("Thinking..."):
                assist_response = await regenerate_response(message)

            full_response = st.write_stream(
                response_generator(assist_response))
            # Add assistant response to chat history
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "faq_id": assist_response.faq_id,
                    "faq_pool_id": assist_response.faq_pool_id,
                    "feedback": None,
                }
            )

    # When user click feedback
    if st.session_state.feedback["is_feedbacked"]:
        await feedback(st.session_state.feedback["feedback"])

    # Accept user input
    if prompt := st.chat_input("Hãy hỏi gì đó đi"):
        st.session_state.feedback = {"is_feedbacked": False, "feedback": None}
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

            # Add user message to chat history
            st.session_state.messages.append(
                {"role": "user", "content": prompt, "faq_id": ""}
            )

        # Display assistant response in chat message container
        with st.chat_message("assistant"):

            message = get_message()

            with st.spinner("Thinking..."):
                assist_response = await send_message(message)

            full_response = st.write_stream(
                response_generator(assist_response))

            st.session_state.messages[-1] = {
                **st.session_state.messages[-1],
                "faq_id": assist_response.faq_id,
            }

            print(st.session_state.messages[-1])
            # Add assistant response to chat history
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "faq_id": assist_response.faq_id,
                    "faq_pool_id": assist_response.faq_pool_id,
                    "feedback": None,
                }
            )

    if (
        len(st.session_state.messages) > 0
        and st.session_state.messages[-1]["faq_id"] is not None
    ):
        with st_horizontal():
            # Regenerate response of assistant
            st.button(
                ":material/replay:", help="Tạo lại câu trả lời", on_click=regenerate
            )

            if not st.session_state.feedback["is_feedbacked"]:
                st.button(
                    ":material/thumb_up_alt:", help="Câu này được đấy", on_click=like
                )
                st.button(
                    ":material/thumb_down_alt:",
                    help="Hmm, còn hơi non",
                    on_click=dislike,
                )


if __name__ == "__main__":
    asyncio.run(main())
