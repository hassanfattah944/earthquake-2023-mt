import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static
st.set_page_config(
    page_title="ML App",
    page_icon="👋",
)

df = pd.read_csv('tweet_data_labeled.csv')
df.dropna(axis=0, inplace=True)##########3###

st.write(df)
def extract_coordinates(coord_string):
    start_index = coord_string.find("longitude=") + len("longitude=")
    end_index = coord_string.find(",", start_index)
    longitude = coord_string[start_index:end_index]

    start_index = coord_string.find("latitude=") + len("latitude=")
    end_index = coord_string.find(")", start_index)
    latitude = coord_string[start_index:end_index]

    return longitude, latitude
extracted_data = df['coordinates'].apply(lambda x: pd.Series(extract_coordinates(x)))
extracted_data.columns = ['longitude', 'latitude']
x = pd.concat([extracted_data], axis=1)
x

model = KMeans(n_init=3)
y_kmeans = model.fit_predict(x)
y_kmeans


x['y'] = y_kmeans
x.head()
st.write(x.head())
st.write(model.inertia_)

wcss = []
for i in range(1,11):
    model = KMeans(n_init=i)
    y_kmeans = model.fit_predict(x)
    wcss.append(model.inertia_)
    
    x = x[:2000]

    cluster1 = x[['longitude', 'latitude']][x['y'] == 0].values.tolist()
    cluster2 = x[['longitude', 'latitude']][x['y'] == 1].values.tolist()
    cluster3 = x[['longitude', 'latitude']][x['y'] == 2].values.tolist()
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=10, tiles="openstreetmap")
    for i in cluster1:
        folium.CircleMarker(i, radius=2, color='blue', fill_color='lightblue').add_to(m)

    for i in cluster2:
        folium.CircleMarker(i, radius=2, color='red', fill_color='lightred').add_to(m)

    for i in cluster3:
        folium.CircleMarker(i, radius=2, color='green', fill_color='lightgreen').add_to(m)

folium_static(m)


hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
