import streamlit as st

st.set_page_config(
    page_title="ML App",
    page_icon="👋",
)


hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
