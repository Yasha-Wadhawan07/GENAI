import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
from data_cleaning import data_for_content_filtering
from scipy.sparse import save_npz, csr_matrix

# Cleaned Data Path
CLEANED_DATA_PATH = "data/cleaned_data.csv"

# cols to transform
numeric_cols = ["danceability", "energy", "key", "loudness", "speechiness", 
               "acousticness", "instrumentalness", "liveness", "valence", 
               "tempo", "duration_ms", "year", "time_signature"]

def train_transformer(data):
    """
    Simplified transformer that handles:
    1. Numeric scaling
    2. TF-IDF for tags
    """
    # Create scalers
    scaler = StandardScaler()
    
    # Scale numeric features
    numeric_features = data[numeric_cols].values
    scaled_features = scaler.fit_transform(numeric_features)
    
    # Create TF-IDF for tags
    tfidf = TfidfVectorizer(max_features=85)
    text_features = tfidf.fit_transform(data['tags'])
    
    # Combine features
    final_features = np.hstack([scaled_features, text_features.toarray()])
    
    # Save transformer components
    joblib.dump({'scaler': scaler, 'tfidf': tfidf}, 'transformer.joblib')
    
    # Save the feature matrix
    np.save('transformed_data.npy', final_features)
    
    return final_features

    # fit the transformer
    transformer.fit(data)

    # save the transformer
    joblib.dump(transformer, "transformer.joblib")
    

def transform_data(data):
    """
    Transforms the input data using the pre-trained transformer components.
    """
    # Load transformer components
    components = joblib.load("transformer.joblib")
    scaler = components['scaler']
    tfidf = components['tfidf']
    
    # Transform numeric features
    numeric_features = data[numeric_cols].values
    scaled_features = scaler.transform(numeric_features)
    
    # Transform text features
    text_features = tfidf.transform(data['tags'])
    
    # Combine features
    final_features = np.hstack([scaled_features, text_features.toarray()])
    
    return final_features

def save_transformed_data(transformed_data, save_path):
    """
    Save the transformed data to a specified file path.
    """
    # Convert to dense array if sparse
    if isinstance(transformed_data, csr_matrix):
        transformed_data = transformed_data.toarray()
    
    # Save as numpy array
    np.save(save_path, transformed_data)

def calculate_similarity_scores(input_vector, data):
    """
    Calculate similarity scores between an input vector and a dataset using cosine similarity.
    """
    return cosine_similarity(input_vector, data)

def content_recommendation(song_name, artist_name, songs_data, transformed_data, k=10):
    """
    Recommends top k songs similar to the given song based on content-based filtering.
    """
    # Convert to lowercase
    song_name = song_name.lower()
    artist_name = artist_name.lower()
    
    # Find the song
    song_row = songs_data.loc[(songs_data["name"] == song_name) & (songs_data["artist"] == artist_name)]
    if len(song_row) == 0:
        return pd.DataFrame(columns=['name', 'artist'])
        
    # Get song index and vector
    song_index = song_row.index[0]
    input_vector = transformed_data[song_index].reshape(1, -1)
    
    # Calculate similarities
    similarity_scores = calculate_similarity_scores(input_vector, transformed_data)
    
    # Get top k similar songs
    top_k_songs_indexes = np.argsort(similarity_scores.ravel())[-k-1:][::-1]
    top_k_songs_names = songs_data.iloc[top_k_songs_indexes]
    
    return top_k_songs_names[['name', 'artist']].reset_index(drop=True)


def main(data_path):
    """
    Test the recommendations for a given song using content-based filtering.
    """
    # Load and clean data
    data = pd.read_csv(data_path)
    
    # Train transformer and transform data
    train_transformer(data)
    transformed_data = np.load('transformed_data.npy')
    
    # Save transformed data
    np.save('data/transformed_data.npy', transformed_data)
    
if __name__ == "__main__":
    main(CLEANED_DATA_PATH)