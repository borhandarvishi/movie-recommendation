from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import pickle

app = Flask(__name__)

# Load the vectorizer
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load the similarity matrix
with open('similarity_matrix.pkl', 'rb') as similarity_file:
    similarity = pickle.load(similarity_file)

# Load the movie dataset
data = pd.read_csv("./movies.csv")
df = pd.DataFrame(data)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # Get the user's input
    movie_name = request.form['movie_name']

    # Finding the close match for the movie name given by the user input
    find_close_match = difflib.get_close_matches(movie_name, df['title'].tolist())
    
    if not find_close_match:
        return render_template('error.html', message="Movie not found. Please try again.")

    close_match = find_close_match[0]
    index_of_movie = df[df.title == close_match]['index'].values[0]
    similarity_score = list(enumerate(similarity[index_of_movie]))

    # Sorting the movies based on their similarity score
    sorted_similarity_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    # Get the top 10 recommendations
    top_recommendations = []
    for movie in sorted_similarity_movies[:10]:
        index = movie[0]
        title_from_index = df[df.index == index]['title'].values[0]
        top_recommendations.append(title_from_index)

    return render_template('recommendations.html', movie_name=movie_name, recommendations=top_recommendations)

if __name__ == '__main__':
    app.run(debug=True)
