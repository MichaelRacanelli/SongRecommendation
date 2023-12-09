import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.spatial.distance import cdist


class Model:
    features = None

    def __init__(self, dataset):
        self.dataset = dataset.sample(frac=0.95, random_state=42)  # Adjust the fraction as needed

    def prepare_model(self):
        # Preprocessing: Label encode genres and normalize tempo
        label_encoder = LabelEncoder()
        genres_encoded = label_encoder.fit_transform(self.dataset['genre'])
        scaler = MinMaxScaler()
        tempo_normalized = scaler.fit_transform(self.dataset[['tempo']])

        # Combine features without one-hot encoding
        self.features = np.hstack((genres_encoded.reshape(-1, 1), tempo_normalized))

   
    # Simplify the recommendation by using a basic similarity metric
    # For example, using Euclidean distance

    # Example usage
    # recommended_songs = recommend_songs('some_song_id', top_n=5)
    def recommend_songs(self, track_id, top_n=10):
        if track_id not in self.dataset['track_id'].values:
            return "Track id not found in the dataset."

        # Find the genre of the song with the given track name
        song_genre = self.dataset[self.dataset['track_id'] == track_id]['genre'].iloc[0]

        # Filter songs that match the genre
        genre_matched_df = self.dataset[self.dataset['genre'] == song_genre]

        # Get the tempo of the input song
        song_tempo = self.dataset[self.dataset['track_id'] == track_id]['tempo'].iloc[0]

        # Compute the absolute difference in tempos
        genre_matched_df['tempo_diff'] = abs(genre_matched_df['tempo'] - song_tempo)

        # Sort by tempo difference and select top_n
        recommendations = genre_matched_df.sort_values('tempo_diff').head(top_n)

        return recommendations[['track_name', 'artist_name', 'genre', 'tempo', 'popularity', 'year', 'danceability', 'energy', 'loudness']]
