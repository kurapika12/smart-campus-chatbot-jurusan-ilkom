import os
import sys

import streamlit as st

# Ensure the project root is on the path so local packages can be imported
sys.path.append(os.path.dirname(__file__))

from controllers.chatbot_controller import (
    get_chatbot_response
)

from views.chat_view import display_chat

# Konfigurasi halaman
st.set_page_config(
    page_title="Smart Campus Chatbot",
    page_icon="🎓",
    layout="centered"
)

# Judul
st.title("AI Assistant Akademik Jurusan Ilmu Komputer")

# Session state
if "messages" not in st.session_state:

    st.session_state.messages = []

# Tampilkan history
for msg in st.session_state.messages:

    display_chat(
        msg["role"],
        msg["content"]
    )

# Input user
user_input = st.chat_input(
    "Tanyakan sesuatu..."
)

# Jika user mengirim pesan
if user_input:

    # Simpan pesan user
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Tampilkan pesan user
    display_chat(
        "user",
        user_input
    )

    # Ambil response chatbot
    response = get_chatbot_response(
        user_input
    )

    # Simpan response chatbot
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    # Tampilkan response chatbot
    display_chat(
        "assistant",
        response
    )
