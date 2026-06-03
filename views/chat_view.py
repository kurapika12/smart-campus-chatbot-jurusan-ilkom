import streamlit as st

def display_chat(role, message):

    with st.chat_message(role):

        st.write(message)