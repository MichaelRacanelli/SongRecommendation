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
        genres_encoded = label_encoder.fit_transform(self.dataset['genre']).reshape(-1, 1)
        scaler = MinMaxScaler()
        tempo_normalized = scaler.fit_transform(self.dataset[['tempo']])

        # Combine features without one-hot encoding
        self.features = np.hstack((genres_encoded, tempo_normalized))

   
    # Simplify the recommendation by using a basic similarity metric
    # For example, using Euclidean distance

    # Example usage
    # recommended_songs = recommend_songs('some_song_id', top_n=5)
    def recommend_songs(self, song_id, top_n=10):
        # Find the index of the song with the given ID
        song_index = self.dataset[self.dataset['track_id'] == song_id].index[0]

        distances = cdist([self.features[song_index]], self.features, 'euclidean')
        top_indices = np.argsort(distances[0])[:top_n]
        return self.dataset.iloc[top_indices]
