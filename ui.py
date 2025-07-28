import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000"

st.title("⚖️ Legal AI Assistant (Tech Law)")

query = st.text_input("Ask a legal question:")
if st.button("Search"):
    resp = requests.post(f"{BACKEND_URL}/search", json={"query": query})
    st.json(resp.json())

if st.button("Ask QA"):
    resp = requests.post(f"{BACKEND_URL}/qa", json={"query": query})
    st.json(resp.json())
