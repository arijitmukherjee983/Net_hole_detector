import streamlit as st
from roboflow import Roboflow

MODEL_ID = "fishing-net-hole-rullc/2"
API_KEY = "Q9X3pzJWmj3VBuDCp0kJ"

@st.cache_resource
def load_model():
    rf = Roboflow(api_key=API_KEY)
    project, version = MODEL_ID.split("/")
    return rf.workspace().project(project).version(int(version)).model
