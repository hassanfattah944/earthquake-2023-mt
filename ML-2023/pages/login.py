import streamlit as st

# Define credentials for admin user
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin"
st.set_page_config(
    page_title="ML App",
    page_icon="👋",
)
def main():
    
    st.title("Admin Dashboard Login")
    st.title("Welcome!")
    link = st.markdown("[Go to other page](hi)")
    username  = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        st.success("Logged in as admin!")
        add_post()
        display_posts()
    elif st.button("Login"):
        st.error("Incorrect username or password")

import streamlit as st
import os
import mysql.connector
from mysql.connector import Error
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

def add_post():
    st.title("Add New Post")
    title = st.text_input("Title")
    content = st.text_area("Content")
    image = st.file_uploader("Upload Image", type=["jpg", "png"])
    if st.button("Create Post"):
        
        if not title:
            st.warning("Please enter a title.")
        elif not content:
            st.warning("Please enter content.")
        elif not image:
            st.warning("Please upload an image.")
        else:
            connection = create_connection()
            if connection:
                cursor = connection.cursor()
                try:
                    cursor.execute("INSERT INTO posts (title, content) VALUES (%s, %s)", (title, content))
                    connection.commit()

                    image_filename = save_image(image, cursor.lastrowid)  # Use the last inserted row ID

                    cursor.execute("UPDATE posts SET image_filename = %s WHERE id = %s", (image_filename, cursor.lastrowid))
                    connection.commit()

                    st.success("Post saved successfully!")
                    
                except Error as e:
                    st.error("Error while saving the post.")
                finally:
                    cursor.close()
                    connection.close()
                    st.write("---------------------------------------------------")
def save_image(image, post_id):
    image_folder = "images"  
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    image_path = os.path.join(image_folder, f"image_{post_id}.jpg")  # Change the file extension as needed
    with open(image_path, "wb") as f:
        f.write(image.read())
    
    return os.path.basename(image_path)  # Return the saved image filename

def display_posts():
    connection = create_connection()
    if connection:
        cursor = connection.cursor(dictionary=True)
        st.write("---------------------------------------------------")
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
                    delete_button = st.button(f"Delete {post['title']}")
                    if delete_button:
                        delete_post(connection, post['id'])
                        st.success(f"Post '{post['title']}' deleted successfully.")

                    st.write("---------------------------------------------------")
        except Error as e:
            st.error("Error while fetching posts.")
        finally:
            cursor.close()
            connection.close()
            st.write("---------------------------------------------------")
def delete_post(connection, post_id):
    cursor = connection.cursor()
    try:
        cursor.execute("DELETE FROM posts WHERE id = %s", (post_id,))
        connection.commit()
    except Error as e:
        st.error("Error while deleting the post.")
    finally:
        cursor.close()
if __name__ == "__main__":
    main()



hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)




    
