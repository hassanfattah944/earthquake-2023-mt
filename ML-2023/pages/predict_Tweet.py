import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
mlp_classifier = MLPClassifier(
    hidden_layer_sizes=(200, 100),  # You can experiment with different architectures
    activation='relu',  # Rectified Linear Unit (ReLU) is a common choice
    solver='adam',  # Adaptive Moment Estimation (Adam) is a good default solver
    alpha=0.0001,  # L2 regularization strength (you can experiment with this)
    learning_rate='adaptive',  # Learning rate adapts during training
    max_iter=1000,  # Maximum number of iterations
    random_state=42  # Set a specific random state for reproducibility
)

st.set_page_config(
    page_title="ML App",
    page_icon="👋",
)
st.title('Tweet Category Prediction')

data = pd.read_csv('tweet_data_labeled.csv')
st.write(data)

label_encoder = LabelEncoder()

encoded_labels = label_encoder.fit_transform(data['Class'])

data['Encoded_Class'] = encoded_labels

unique_numerical_labels = label_encoder.classes_
for numerical_label, class_label in enumerate(unique_numerical_labels):
    print(f"Numerical Label: {numerical_label}, Class Label: {class_label}")

X = data['cleaned_content']
y = data['Encoded_Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MLPClassifier())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
tweet = st.text_area('Enter a Tweet')

if st.button('Predict'):
    if not tweet:
        st.warning('Please enter a tweet to predict.')
    else:
        prediction = pipeline.predict([tweet])[0]
        predicted_class = label_encoder.inverse_transform([prediction])[0]
        st.success(f'Predicted Class: {predicted_class}')
        st.success(f"Accuracy of this Models is: {accuracy:.2f}")
        

hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

