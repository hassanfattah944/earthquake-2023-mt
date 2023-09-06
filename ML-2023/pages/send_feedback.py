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
    
    # Load existing feedback data (if any)
    try:
        existing_feedback_data = pd.read_csv(feedback_file)
    except FileNotFoundError:
        existing_feedback_data = pd.DataFrame(columns=["Timestamp", "Name", "Feedback", "Rating"])
    
    # Append the new feedback and save to Excel
    updated_feedback_data = existing_feedback_data.append(feedback_data, ignore_index=True)
    updated_feedback_data.to_csv(feedback_file, index=False)

def main():
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
    try:
        feedback_data = pd.read_csv(feedback_file)
        if not feedback_data.empty:
            st.dataframe(feedback_data)
        else:
            st.info("No feedback yet.")
    except FileNotFoundError:
        st.info("No feedback yet. Be the first to provide feedback!")

if __name__ == "__main__":
    main()
