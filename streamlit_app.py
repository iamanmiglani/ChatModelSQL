class StreamlitChatBot:
    def __init__(self):
        # Initialize session state for OpenAI API key
        if 'openai_api_key' not in st.session_state:
            st.session_state.openai_api_key = None
        if 'chatbot' not in st.session_state:
            st.session_state.chatbot = None
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'current_table' not in st.session_state:
            st.session_state.current_table = None

    def setup_page(self):
        """Set up the Streamlit page layout."""
        st.set_page_config(
            page_title="AI ChatBot with Data Management",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        st.title("AI-Powered ChatBot with Data Management")

    def render_sidebar(self):
        """Render the sidebar with API key input, data upload, and table selection options."""
        with st.sidebar:
            st.header("Settings")

            # Request OpenAI API key from user
            st.session_state.openai_api_key = st.text_input(
                "Enter OpenAI API Key",
                type="password",
                help="Your OpenAI API key will be used only during this session and not stored."
            )

            if st.session_state.openai_api_key:
                st.success("API Key Set!")
                # Reinitialize chatbot with the new API key if it hasn't been initialized
                if st.session_state.chatbot is None:
                    st.session_state.chatbot = setup_chatbot(
                        api_key=st.session_state.openai_api_key
                    )

            else:
                st.warning("Please enter your OpenAI API key to proceed.")

            st.header("Data Management")
            uploaded_file = st.file_uploader(
                "Upload Data File",
                type=['csv', 'xlsx', 'xls', 'db'],
                help="Upload a CSV, Excel, or SQLite database file"
            )
            
            # Rest of the data management code remains the same
            ...

    def render_chat_interface(self):
        """Render the main chat interface."""
        st.header("Chat Interface")
        
        if not st.session_state.openai_api_key:
            st.warning("Please enter your OpenAI API key in the sidebar to use the chatbot.")
            return
        
        user_input = st.text_input("Ask a question:")
        
        if user_input and st.session_state.chatbot:
            response = st.session_state.chatbot.process_question(user_input)
            if "error" in response:
                st.error(f"Error: {response['error']}")
            else:
                st.write("### Query:")
                st.code(response["query"])
                st.write("### Results:")
                st.dataframe(pd.DataFrame(response["data"]))
                st.write("### Summary:")
                st.info(response["summary"])
                
                st.session_state.chat_history.append({
                    "question": user_input,
                    "response": response
                })

        # Display chat history
        if st.session_state.chat_history:
            st.subheader("Chat History")
            for i, entry in enumerate(st.session_state.chat_history):
                st.write(f"**Q{i+1}: {entry['question']}**")
                st.write(f"**A{i+1}: {entry['response']['summary']}**")

# Modify the `setup_chatbot` function to accept the API key
def setup_chatbot(api_key: str) -> ChatBot:
    """Setup and initialize the chatbot with the provided API key."""
    df_manager = DataFrameManager()
    query_generator = QueryGenerator(df_manager, api_key)
    return ChatBot(df_manager, query_generator)
