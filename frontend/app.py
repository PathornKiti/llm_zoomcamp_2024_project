import streamlit as st
import requests
import uuid  
from langfuse import Langfuse
import os

# Initialize Langfuse
langfuse = Langfuse(
    secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
    public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
    host=os.environ.get("LANGFUSE_HOST")
)

# Set up the Streamlit page
st.set_page_config(page_title="Relationship Counselor Chatbot", page_icon="💬")

# Generate a unique session ID and store it in the session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Initialize chat history if not already present
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# Track if feedback was given for the last bot message
if 'last_feedback_given' not in st.session_state:
    st.session_state.last_feedback_given = False

# Define the feedback processing function
def process_feedback(thumbs_up, response):
    comment = "Thumbs-up" if thumbs_up else "Thumbs-down"
    rating = 1 if thumbs_up else 0
    # Send feedback to Langfuse
    langfuse.score(
        trace_id=st.session_state.session_id, 
        name="user_feedback", 
        comment=comment,
        value=rating
    )
    st.success(f"Feedback submitted: {comment}")
    st.session_state.last_feedback_given = True  # Mark feedback as given

# Function to interact with the FastAPI backend
def get_response_from_backend(user_input):
    try:
        # Send the user's input to the backend API
        response = requests.post(
            "http://backend:8000/query", 
            json={
                "input": user_input, 
                "session_id": st.session_state.session_id  # Send the session UUID
            }
        )
        
        # If response is successful
        if response.status_code == 200:
            data = response.json()
            return data.get('response', 'I could not understand your query.')
        else:
            return "Error: Unable to get a response from the counselor at the moment."
    except Exception as e:
        return f"Error: {str(e)}"

# Display the conversation history using st.chat_message
for i, message in enumerate(st.session_state.conversation):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# If a new message is entered, the previous feedback buttons will disappear
if st.session_state.last_feedback_given:
    st.session_state.last_feedback_given = False  # Reset feedback for new conversation

# User input and handling
if prompt := st.chat_input("Feel free to share anything on your mind about relationships. I'm here to listen and offer support whenever you need it."):
    # Display user message in chat container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to conversation history
    st.session_state.conversation.append({"role": "user", "content": prompt})
    
    # Get response from the backend
    response = get_response_from_backend(prompt)
    
    # Display bot response in chat container
    with st.chat_message("bot"):
        st.markdown(response)
    
    # Add bot message to conversation history
    st.session_state.conversation.append({"role": "bot", "content": response})

# Show feedback buttons for the most recent bot response if feedback hasn't been given yet
if not st.session_state.last_feedback_given and st.session_state.conversation and st.session_state.conversation[-1]['role'] == 'bot':
    thumbs_up, thumbs_down = st.columns([1, 1])
    with thumbs_up:
        if st.button('👍', key=f"thumbsup_{len(st.session_state.conversation)}"):
            process_feedback(True, st.session_state.conversation[-1]['content'])
    with thumbs_down:
        if st.button('👎', key=f"thumbsdown_{len(st.session_state.conversation)}"):
            process_feedback(False, st.session_state.conversation[-1]['content'])

# Footer
st.markdown("---")
st.markdown("👨‍💻 Developed by [Pathorn Kitisupavatana](https://github.com/PathornKiti)")
st.markdown("For relationship advice and support.")
