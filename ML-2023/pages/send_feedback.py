import streamlit as st
import pandas as pd
from datetime import datetime
st.set_page_config(
    page_title="ML App",
    page_icon="👋",
)
feedback_file = "feedback.csv"

def save_feedback(name, feedback, rating):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    feedback_data = pd.DataFrame({"Timestamp": [timestamp], "Name": [name], "Feedback": [feedback], "Rating": [rating]})
    
    if not st.session_state.feedback_df:
        st.session_state.feedback_df = pd.DataFrame(columns=["Timestamp", "Name", "Feedback", "Rating"])
        
    st.session_state.feedback_df = st.session_state.feedback_df.append(feedback_data, ignore_index=True)
    st.session_state.feedback_df.to_csv(feedback_file, index=False)

def main():

    if "feedback_df" not in st.session_state:
        st.session_state.feedback_df = None
    st.header("Provide Feedback")
    name = st.text_input("Your Name")
    feedback = st.text_area("Your Feedback")
    rating = None
    if st.button("👍"):
        rating = "👍"
    if st.button("👎"):
        rating = "👎"
    if name and feedback and rating:
        save_feedback(name, feedback, rating)
        st.success("Thank you for your feedback!")
    st.header("Saved Feedback")
    if st.session_state.feedback_df is not None and not st.session_state.feedback_df.empty:
        st.dataframe(st.session_state.feedback_df)
    else:
        st.info("No feedback yet.")

if __name__ == "__main__":
    main()
hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
