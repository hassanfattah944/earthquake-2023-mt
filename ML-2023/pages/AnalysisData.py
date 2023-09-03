import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
import streamlit as st
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


st.set_page_config(
    page_title="ML App",
    page_icon="👋",
)


st.subheader("ML Case-Studies | Natural Language Processing (NLP).")
st.error('A. Data Analysis (Earthquake ) :')
data = pd.read_csv('ML-2023/first_40000_rows.csv', nrows=15000)
st.info('Importing Datasets.')

st.write(data)
dataset_info = [
    f"Number of rows: {len(data)}",
    f"Number of columns: {len(data.columns)}",
    "Column names: " + ", ".join(data.columns),
    "Data types:\n" + "\n".join([f"{col}: {dtype}" for col, dtype in data.dtypes.items()])
]
st.info("Dataset Information:")
st.table(dataset_info)
st.info('Display all the columns in the DataFrame.')
st.write(data.columns)
st.info('Check the unique Source name where the mostly tweet are Submit.')
st.write(data['source'].value_counts())
st.info('Checks if there are any duplicates, All row returned False Which means there are no Duplicates.')
st.write(data.duplicated())
st.info('Calling the count() method on the dataframe: returns the sum of all entries in a column.')
st.write(data.count())
st.info('Let''s Observe whether null values are present.')
st.write(data.isnull().any())
st.markdown('Exploring Data.')
st.write(data['content'].apply(len).max())
st.info('checking size of data.')
data.shape
st.info('Check for duplicate records.')
duplicate_rows = data.duplicated()
data[duplicate_rows]
st.info('Remove null values.')
st.error('B.Data Preprocessing.')
def filter_by_language(dataset, language):
    return dataset[data['language'] == language]

filtered_data  = filter_by_language(data, 'en')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define a function for text cleaning and preprocessing
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove mentions (e.g., @username)
    text = re.sub(r'@\w+', '', text)

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Convert to lowercase
    text = text.lower()

    # Tokenize the text into words
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]

    # Stemming (reducing words to their root form)
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]

    # Join the cleaned and preprocessed words back into a sentence
    cleaned_text = ' '.join(stemmed_words)


    return cleaned_text

# Apply the preprocessing function to the 'cleaned_content' column
filtered_data['cleaned_content'] = filtered_data['content'].apply(preprocess_text)

import demoji
demoji.download_codes()
def processed_text(text):
    # Remove emojis using the demoji library
    text = demoji.replace(text, repl="")
    # Remove numbers using regex
    text = re.sub(r'\d', '', text)
    return text
st.info('Clean The DataSet.')
# Apply the preprocessing function to the 'cleaned_content' column
st.write(filtered_data)
st.info('Printing the data to see the effect of preprocessing+NLP.')
st.write(filtered_data['cleaned_content'].iloc[0],"\n")
st.write(filtered_data['cleaned_content'].iloc[1],"\n")
st.write(filtered_data['cleaned_content'].iloc[2],"\n")
st.write(filtered_data['cleaned_content'].iloc[3],"\n")
st.write(filtered_data['cleaned_content'].iloc[4],"\n")

st.info('Check the size of the cleaned dataset')
filtered_data.shape





classes = {
    'Help': ['help', 'assistance', 'support', 'donate'],
    'News': ['earthquake', 'magnitude', 'aftershock', 'rescue', 'recovery'],
    'Funding': ['fundraiser', 'donation', 'charity', 'fund'],
    'Survivor stories': ['survivor', 'personal account', 'experience', 'impact'],
    'Volunteering': ['volunteer', 'helping', 'aid', 'support'],
    'Political response': ['government', 'political', 'action', 'response'],
    'Humanitarian aid': ['humanitarian', 'aid', 'relief', 'support'],
    'Casualty reports': ['injury', 'death', 'fatal', 'missing', 'victims'],
    'Infrastructure damage': ['building', 'bridge', 'road', 'power', 'water', 'gas', 'telecommunications'],
    'Weather conditions': ['storm', 'rain', 'snow', 'wind', 'temperature', 'weather'],
    'Emergency services': ['ambulance', 'fire', 'police', 'emergency', 'rescue'],
    'Social media activity': ['tweet', 'post', 'share', 'social media'],
    'Prayer and condolences': ['prayer', 'thoughts', 'condolences', 'sympathy'],
    'Technical information': ['data', 'statistics', 'analysis', 'technical'],
    'International aid': ['international', 'donor', 'aid', 'relief'],
    'Business impact': ['business', 'economic', 'financial', 'impact'],
    'Personal safety': ['safety', 'evacuation', 'shelter', 'protection', 'precaution']
}
# # Define a function to label each tweet based on its content
# def classify_tweet(tweet_text):
#     for class_name, keywords in classes.items():
#         for keyword in keywords:
#             if keyword in tweet_text:
#                 return class_name
#     return 'Other'  # If no keyword matches, label as "Other"
def expand_keywords(keywords):
    expanded_keywords = set(keywords)
    for keyword in keywords:
        synsets = wordnet.synsets(keyword)
        for synset in synsets:
            for lemma in synset.lemmas():
                expanded_keywords.add(lemma.name())
    return list(expanded_keywords)

# Apply the expand_keywords function to each label's keywords
for label in classes:
    classes[label] = expand_keywords(classes[label])

# Define the assign_label function with expanded keywords
def assign_label(sentence):
    for label, keywords in classes.items():
        for keyword in keywords:
            if keyword in sentence:
                return label
    return "Uncategorized"

# Add a new column to the dataframe with the class label for each tweet
filtered_data['Class'] = filtered_data['cleaned_content'].apply(assign_label)

# Save the labeled dataset to a new CSV file
filtered_data.to_csv('tweet_data_labeled.csv', index=False)

df = pd.read_csv('tweet_data_labeled.csv')
from sklearn.preprocessing import LabelEncoder

# Create a label encoder instance
label_encoder = LabelEncoder()

# Fit the label encoder to your class labels
encoded_labels = label_encoder.fit_transform(df['Class'])

# Add the encoded labels to your dataframe
df['Encoded_Class'] = encoded_labels

# Print the first few rows of the dataframe with encoded labels
print(df[['cleaned_content', 'Class', 'Encoded_Class']].head())




#  df is your DataFrame containing the 'Class' column

# Create a bar plot using Matplotlib
st.error("C.Exploratory Data Analysis(EDA).")



st.info('Class Distribution of Tweet Data')
fig, ax = plt.subplots()
class_counts = df['Class'].value_counts()
class_counts.plot(kind='bar')
#plt.title('Class Distribution of Tweet Data')
plt.xlabel('Class')
plt.ylabel('Number of Tweets')
# Display the Matplotlib plot using Streamlit's pyplot function
st.pyplot(plt)
 
from wordcloud import WordCloud
st.info("Word Cloud Example")
fig2=plt.figure(figsize=(20,15))
text = ' '.join([word for word in df['content']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words used in  Tweets', fontsize=19)

# Streamlit app code

st.image(wordcloud.to_array(), use_column_width=True)
 # Display the word cloud using Streamlit's pyplot function

st.error("D.Encoding Data.")
from sklearn.preprocessing import LabelEncoder
# Create a label encoder instance
label_encoder = LabelEncoder()
# Fit the label encoder to your class labels
encoded_labels = label_encoder.fit_transform(df['Class'])

# Add the encoded labels to your dataframe
df['Encoded_Class'] = encoded_labels

# Print the first few rows of the dataframe with encoded labels
st.write(df[['cleaned_content', 'Class', 'Encoded_Class']].head())



num_unique_labels = len(df['Class'].unique())
st.write("Number of unique labels:", num_unique_labels)


# Get the unique numerical labels
unique_numerical_labels = label_encoder.classes_

# 
st.info('Print the mapping of numerical labels to class labels:')
# Create a list of dictionaries to represent the data
data = []
for numerical_label, class_label in enumerate(unique_numerical_labels):
    data.append({"Numerical Label": numerical_label, "Class Label": class_label})
st.table(data)


class_counts = df['Class'].value_counts()
st.info('Count the number of instances in each class:')

# Print the class distribution
print('Class distribution:')
st.write(class_counts)



st.error("E.Models Building.")
st.info("Feature extraction algorithm.")
y=df['Encoded_Class']
X=df['cleaned_content']
st.write(y)
st.write(X)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
# Fit the vectorizer on the training data
X=vectorizer.fit_transform(X)
st.info("Create a TF-IDF vectorizer to convert text to numerical features.")
st.write("Size of vectorizer on the training data",X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
st.info("Split the dataset into training and testing sets,80% for training and 20% for  test.")
st.write("Size of x_train is:", (X_train.shape))
st.write("Size of x_test  is: ", (X_test.shape))

from sklearn.svm import LinearSVC
#Train a Support Vector Machine model on the training data
svm_model = LinearSVC()
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
d = dt.predict(X_test)
DecisionTree_accuracy=accuracy_score(y_test, d)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)

from sklearn.naive_bayes import MultinomialNB
NaiveBayes_model = MultinomialNB()
NaiveBayes_model.fit(X_train, y_train)
NaiveBayes_pred= NaiveBayes_model.predict(X_test)
NaiveBayes_accuracy=accuracy_score(y_test, NaiveBayes_pred)
st.error("F.Evaluation Models.")
st.info("Print the accuracy of both models.")
st.write("Support Vector Machine Accuracy is:{:.2f}%".format(svm_accuracy*100))
st.write("Decision Tree Classifier is:{:.2f}%".format(DecisionTree_accuracy*100))
st.write("Logistic Regression Accuracy is:{:.2f}%".format(lr_accuracy*100))
st.write("Multinomial Naive Bayes is:{:.2f}%".format(NaiveBayes_accuracy*100))


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initialize the MLP classifier with adjusted hyperparameters
mlp_classifier = MLPClassifier(
    hidden_layer_sizes=(200, 100),  # You can experiment with different architectures
    activation='relu',  # Rectified Linear Unit (ReLU) is a common choice
    solver='adam',  # Adaptive Moment Estimation (Adam) is a good default solver
    alpha=0.0001,  # L2 regularization strength (you can experiment with this)
    learning_rate='adaptive',  # Learning rate adapts during training
    max_iter=1000,  # Maximum number of iterations
    random_state=42  # Set a specific random state for reproducibility
)

# Train the MLP classifier on the training data
mlp_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_test_predicted_mlp = mlp_classifier.predict(X_test)
pred_prob_mlp = mlp_classifier.predict_proba(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_test_predicted_mlp)
st.write("Accuracy of  Neural Network: {:.2f}%".format(accuracy*100))




st.info("Display classification Report Of Neural Network.")
# Generate classification report
class_report = classification_report(y_test, y_test_predicted_mlp)
st.write("Classification Report:\n", class_report)


st.info("Confusion Matrix Of Support Vector Machine.")
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import style
style.use('ggplot')
style.use('classic')
cm = confusion_matrix(y_test, svm_pred, labels=svm_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_model.classes_)
# Plot the confusion matrix
fig1, ax = plt.subplots()
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig1)
st.info("Confusion Matrix Of Decision Tree Classifier.")
style.use('ggplot')
style.use('classic')
cm = confusion_matrix(y_test, d, labels=dt.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt.classes_)
# Plot the confusion matrix
fig2, ax = plt.subplots()
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig2)
st.info("Confusion Matrix Of Logistic Regression.")
style.use('ggplot')
style.use('classic')
cm = confusion_matrix(y_test, lr_pred, labels=lr_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lr_model.classes_)
# Plot the confusion matrix
fig3, ax = plt.subplots()
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig3)
st.info("Confusion Matrix Of Multinomial Naive Bayes.")
style.use('ggplot')
style.use('classic')
cm = confusion_matrix(y_test, NaiveBayes_pred, labels=NaiveBayes_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=NaiveBayes_model.classes_)
# Plot the confusion matrix
fig4, ax = plt.subplots()
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig4)
st.info("Confusion Matrix Of Neural Network.")
style.use('ggplot')
style.use('classic')
cm = confusion_matrix(y_test, y_test_predicted_mlp, labels=mlp_classifier.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp_classifier.classes_)
# Plot the confusion matrix
fig5, ax = plt.subplots()
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig5)
hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
