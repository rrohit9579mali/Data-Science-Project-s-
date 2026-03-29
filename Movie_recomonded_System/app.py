import streamlit as st
import pandas as pd
import pickle
import requests
import io
import os
# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title='Movie Recommendation System', layout='wide')

# ---------------- UI HEADER ----------------
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)

# ---------------- LOAD DATASET ----------------
# Ensure your CSV is in the Datasets folder on GitHub
# data = pd.read_csv('Datasets/final_df_movie_recommend_system.csv')
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, 'Datasets', 'final_df_movie_recommend_system.csv')
data = pd.read_csv(csv_path)
# ---------------- FETCH MODEL INTO VARIABLE (NO DISK WRITE) ----------------
@st.cache_resource
def load_sim_score(url):
    local_filename = "temp_sim_score.pkl"
    if not os.path.exists(local_filename):
        with st.spinner("Downloading model... 🚀"):
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(local_filename, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                st.error("Download failed!")
                return None
    return pickle.load(open(local_filename, 'rb'))

# Your Hugging Face Resolve Link
HF_URL ="https://huggingface.co/datasets/Rohit-Mali-2005/movies_data/resolve/main/sim_score.pkl"
sim_score = load_sim_score(HF_URL)

if sim_score is None:
    st.stop()

# ---------------- FETCH POSTER FUNCTION ----------------
def fetch_poster(movie_id):
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US'
    try:
        response = requests.get(url)
        data_json = response.json()
        if 'poster_path' in data_json and data_json['poster_path']:
            return "https://image.tmdb.org/t/p/w500/" + data_json['poster_path']
    except:
        pass
    return "https://via.placeholder.com/500x750?text=No+Image"

# ---------------- RECOMMEND FUNCTION ----------------
def recommend(movie):
    movie_index = data[data['title'] == movie].index[0]
    distances = sim_score[movie_index] # Directly using the variable
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:10]
    return [i[0] for i in movies_list]

# ---------------- INTERACTIVE UI ----------------
title = sorted(data['title'])
selected_movie = st.selectbox('Select Movie Title', title)

if st.button('🚀 Recommend'):
    indices = recommend(selected_movie)
    recommended_movies = data.iloc[indices]

    st.subheader('Recommended Movies 🎥')
    n = 3  # movies per row
    for i in range(0, len(recommended_movies), n):
        cols = st.columns(n)
        for j, idx in enumerate(range(i, i + n)):
            if idx < len(recommended_movies):
                movie = recommended_movies.iloc[idx]
                poster_url = fetch_poster(movie['id'])
                with cols[j]:
                    st.image(poster_url, use_container_width=True)
                    st.markdown(f"**{movie['title']}**")
                    st.caption(f"ID: {movie['id']}")
