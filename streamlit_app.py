import streamlit as st
import pandas as pd
from chatbot import setup_chatbot

class StreamlitChatBot:
    def __init__(self):
        if 'openai_api_key' not in st.session_state:
            st.session_state.openai_api_key = None
        if 'chatbot' not in st.session_state:
            st.session_state.chatbot = None

    def setup_page(self):
        st.set_page_config(page_title="AI Chatbot", layout="wide")
        st.title("AI Chatbot")

    def render_sidebar(self):
        with st.sidebar:
            st.header("Settings")
            st.session_state.openai_api_key = st.text_input("Enter OpenAI API Key", type="password")
            if st.session_state.openai_api_key:
                st.success("API Key Set!")

    def render_chat_interface(self):
        st.header("Chat Interface")
        if st.session_state.openai_api_key and st.session_state.chatbot is None:
            st.session_state.chatbot = setup_chatbot(st.session_state.openai_api_key)

        user_input = st.text_input("Ask a question:")
        if user_input and st.session_state.chatbot:
            response = st.session_state.chatbot.process_question(user_input)
            st.write("### Query:", response.get("query", ""))
            st.write("### Results:", pd.DataFrame(response.get("data", [])))
            st.write("### Summary:", response.get("summary", ""))


def main():
    app = StreamlitChatBot()
    app.setup_page()
    app.render_sidebar()
    app.render_chat_interface()


if __name__ == "__main__":
    main()
