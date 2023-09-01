import streamlit as st
import mysql.connector
from mysql.connector import Error
import os
st.set_page_config(
    page_title="ML App",
    page_icon="👋",
)
# Function to connect to the MySQL database
def create_connection():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database='earthquake-2023-mt',
            user='root',
            password=''
        )
        
        if connection.is_connected():
            return connection
    except Error as e:
        print("Error while connecting to MySQL:", e)
        return None

# Define function to fetch and display posts
def display_posts():
    connection = create_connection()
    if connection:
        cursor = connection.cursor(dictionary=True)
        try:
            cursor.execute("SELECT * FROM posts")
            posts = cursor.fetchall()

            if not posts:
                st.info("No posts available.")
            else:
                for post in posts:
                    st.title(post['title'])
                    st.write(post['content'])
                    image_path = os.path.join("images", post['image_filename'])
                    st.image(image_path, caption=post['title'], use_column_width=True)
                    st.write("---------------------------------------------------")
        except Error as e:
            st.error("Error while fetching posts.")
        finally:
            cursor.close()
            connection.close()


# Call the main function to start the Streamlit app
if __name__ == "__main__":
    display_posts()


hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
