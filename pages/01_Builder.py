import streamlit as st
import importlib

st.set_page_config(page_title="Rollercoaster Builder", layout="wide")

st.title("Builder")
st.caption("Design, simulate, and visualize coaster tracks.")

# Importing app_builder will execute its Streamlit UI at module top-level
try:
    import app_builder  # noqa: F401
    importlib.reload(app_builder)
except Exception as e:
    st.error(f"Failed to load builder UI: {e}")
