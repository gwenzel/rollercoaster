import streamlit as st

st.set_page_config(page_title="Rollercoaster", layout="wide")

st.title("Welcome to Rollercoaster")
st.write("Use the left sidebar to open pages.")
st.markdown("- Builder: design and simulate tracks\n- RFDB Data: analyze real ride datasets")
st.success("Tip: Run the app with `streamlit run app.py`.")
