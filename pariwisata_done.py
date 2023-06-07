import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import streamlit as st


from sklearn.cluster import KMeans
import numpy as np 

import seaborn as sns
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.metrics.pairwise import cosine_similarity

from zipfile import ZipFile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

def main():
    st.title("Sistem Rekomendasi Content Based Filtering")
    
    tv = TfidfVectorizer(max_features=5000)
    stem = StemmerFactory().create_stemmer()
    stopword = StopWordRemoverFactory().create_stop_word_remover()

    data_tourism_rating = pd.read_csv('tourism_rating.csv')
    data_tourism_with_id = pd.read_csv('tourism_with_id.csv')
    data_user = pd.read_csv('user.csv')

    # st.write(data_tourism_rating.head())
    # st.write(data_tourism_with_id.head())
    # st.write(data_user.head())

    data_tourism_with_id.drop(['Rating','Time_Minutes','Coordinate','Lat','Long','Unnamed: 11','Unnamed: 12'],axis=1,inplace=True)

    # st.write("-----")
    # st.write(" Data Tourism With ID ")
    # st.write(data_tourism_with_id)

    data_rekomendasi = pd.merge(data_tourism_rating.groupby('Place_Id')['Place_Ratings'].mean(),data_tourism_with_id,on='Place_Id')
    # st.write("------")
    # st.write(" Data Rekomendasi ")
    # st.write(data_rekomendasi)

    def preprocessing(data):
        data = data.lower()
        data = stem.stem(data)
        data = stopword.remove(data)
        return data

    data_content_based_filtering = data_rekomendasi.copy()
    data_content_based_filtering['Tags'] = data_content_based_filtering['Description'] + ' ' + data_content_based_filtering['Category']
    data_content_based_filtering.drop(['Price','Place_Ratings','Description','Category'],axis=1,inplace=True)
    # st.write("----")
    # st.write("Data Filltering")
    # st.write(data_content_based_filtering.head())

    data_content_based_filtering.Tags = data_content_based_filtering.Tags.apply(preprocessing)
    # st.write("----")
    # st.write("Data Content Based")
    # st.write(data_content_based_filtering.head())

    # Mengambil kolom 'Tags' untuk preprocessing dan train data
    data_train = data_content_based_filtering[['Tags']].copy()

    # Preprocessing
    data_train['Tags'] = data_train['Tags'].apply(preprocessing)

    # Melakukan fit transform pada data menggunakan TfidfVectorizer
    vectors = tv.fit_transform(data_train['Tags']).toarray()

    # Melakukan cosine similarity
    similarity = cosine_similarity(vectors)

    # Menyimpan hasil similarity
    np.save('similarity.npy', similarity)

    # st.write("----- Hasil Train -----")
    # st.write("Data Train:")
    # st.write(data_train.head())

    # st.write("\nVektors:")
    # st.write(vectors)

    # st.write("\nSimilarity Matrix:")
    # st.write(similarity)

    # st.write("---------------")

    def recommend_by_content_based_filtering(nama_tempat):
        nama_tempat_index = data_content_based_filtering[data_content_based_filtering['City'] == nama_tempat].index[0]
        distancess = similarity[nama_tempat_index]
        nama_tempat_list = sorted(list(enumerate(distancess)), key=lambda x: x[1], reverse=True)[1:20]

        recommended_nama_tempats = []
        for i in nama_tempat_list:
            recommended_nama_tempats.append(([data_content_based_filtering.iloc[i[0]].Place_Name]+[i[1]]))

        return recommended_nama_tempats

    def print_recommendations(recommendations):
        st.write("------ Hasil Rekomendasi Tempat Berdasarkan Kota Jakarta ------")
        col1, col2 = st.columns(2)  # Mengatur jumlah kolom menjadi 2

        for index, recommendation in enumerate(recommendations, start=1):
            place_name, similarity_score = recommendation
            if index % 2 == 1:
                with col1:
                    st.write(f"{index}. {place_name}, Similarity Score: {similarity_score}")
            else:
                 with col2:
                    st.write(f"{index}. {place_name}, Similarity Score: {similarity_score}")

    city_name = 'Jakarta'
    recommendations = recommend_by_content_based_filtering(city_name)
    print_recommendations(recommendations)

    st.write("---------------")
    
    hasil_rekomendasi = np.array(recommendations)

    def perform_clustering(hasil_rekomendasi):
        features = hasil_rekomendasi[:, 1].astype(float)
        features = features.reshape(-1, 1)

        num_clusters = 3

        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(features)

        cluster_labels = kmeans.labels_

        clustered_hasil_rekomendasi = []
        for i, item in enumerate(hasil_rekomendasi):
            item_cluster = cluster_labels[i]
            clustered_hasil_rekomendasi.append([item[0], item_cluster])

        temp = []
        for item in clustered_hasil_rekomendasi:
            temp.append(item)

        hasil_cluster = np.array(temp)

        features = hasil_rekomendasi[:, 1].astype(float)
        features = features.reshape(-1, 1)

        k_values = range(1, len(hasil_rekomendasi) + 1)
        inertia_values = []

        for k in k_values:
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(features)
            inertia_values.append(kmeans.inertia_)

    # Plot elbow curve
        st.write("Grafik Elbow Berdasarkan Hasil Clustering")
        fig, ax = plt.subplots()
        ax.plot(list(k_values), inertia_values, 'bx-')
        ax.set(xlabel='Number of Clusters (k)', ylabel='Inertia', title='Elbow Curve')
        st.pyplot(fig)

        return hasil_cluster
    
    st.write(" Hasil Clustering ")
    clustered_hasil_rekomendasi = perform_clustering(hasil_rekomendasi)
    st.write(clustered_hasil_rekomendasi)

    
# Menjalankan aplikasi Streamlit
if __name__ == '__main__':
    main()
