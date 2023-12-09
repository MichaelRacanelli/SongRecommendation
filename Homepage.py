import streamlit as st
import pandas as pd
from PIL import Image
from search_fuzzy import find_top_matches
from model import Model
import zipfile

image = Image.open('spotify.jpg')

# Create an about widget
with st.sidebar:
    st.title("We are Group 12 from WatSpeed")
    st.text("And this is our song recommendation app")
    st.image(image, caption='Spotify Genres')
    st.header("Image Credit")
    st.text("Photo by David Pupăză Unsplash")

# Specify the path to your ZIP file
zip_file_path = "spotify_data.zip"

@st.cache_data()
def load_data(zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
        csv_file_name = next(name for name in zip_file.namelist() if name.endswith('.csv'))
        spotify_data = pd.read_csv(zip_file.open(csv_file_name), index_col=0)
    dtype_mapping = {
    "artist_name": "string",
    "track_name": "string",
    "track_id": "string",
    "popularity": "int",
    "year": "int",
    "genre": "string",
    "danceability": "float",
    "energy": "float",
    "key": "int",
    "loudness": "float",
    "mode": "int",
    "speechiness": "float",
    "acousticness": "float",
    "instrumentalness": "float",
    "liveness": "float",
    "valence": "float",
    "tempo": "float",
    "duration_ms": "float",
    "time_signature": "int"
    }

    spotify_data = spotify_data.astype(dtype_mapping)

    # Assuming the numerical columns are the ones that need outlier removal
    numerical_columns = spotify_data.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Function to remove outliers using IQR method
    def remove_outliers(df, columns):
        for column in columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        return df

    # Remove outliers
    spotify_data_cleaned = remove_outliers(spotify_data, numerical_columns)
    return spotify_data_cleaned

# Call the data loading function
spotify_data_cleaned = load_data("spotify_data.zip")

@st.cache_data()
def prepare_model(data):
    model = Model(data)
    model.prepare_model()
    return model

model = prepare_model(spotify_data_cleaned)

def perform_fuzzy_matching(user_input, data, artist_name=None):
    # Filter data based on the optional artist name
    if artist_name:
        filtered_data = data[data['artist_name'].str.contains(artist_name, case=False, na=False)]
    else:
        filtered_data = data
    
    return find_top_matches(user_input, filtered_data)

# Streamlit app title header
st.title('Song Finder')

# Widget for user input (song name)
user_input_song = st.text_input('Type in a song name:', value='', key='song_input', help='Fuzzy Matching will generate the closest matches')

# Placeholder for selected track ID
selected_track_id = None

# Actions to be performed when the user inputs a song in the search box
if user_input_song:
    # Optional: Widget for user input (artist name)
    user_input_artist = st.text_input('Optional: Type in an artist name:', value='', key='artist_input', help='Type in an artist if you need to narrow down the search results')
    
    # Generate fuzzy matching top results for search input
    top_results = perform_fuzzy_matching(user_input_song, spotify_data_cleaned, artist_name=user_input_artist)

    # Create a list item of clean song, artist, and similarity score for displaying in the selectbox widget
    clean_results = [f"'{result['original_track_name']}' by {result['artist']} | Similarity Score: {result['similarity_score']}" for result in top_results]

    # Display the clean search results as a clickable list with both song and artist names with similarity score
    st.subheader('Select a song:')
    selected_song_index = st.selectbox('Choose the song you are looking for from the closest results:', 
                                    range(len(clean_results)),  # Use range(len(clean_results)) as the options
                                    format_func=lambda x: clean_results[x],  # Display clean_results in the dropdown
                                    index=0, 
                                    key='selected_song')
    
    # Extract the track ID from the selected result
    if selected_song_index is not None:
        selected_track_id = top_results[selected_song_index]['track_id']
        # Display information about the selected song using extracted song ID
        selected_song_info = spotify_data_cleaned[spotify_data_cleaned['track_id'] == selected_track_id]
        st.write(selected_song_info[['track_name', 'artist_name', 'genre']].squeeze())

# Buttons to generate and clear recommendations
if selected_track_id is not None:
    if st.button('Generate Recommendations'):
        recommended_songs_df = model.recommend_songs(selected_track_id)
        st.write(recommended_songs_df[['track_name', 'artist_name', 'genre']].squeeze())

    if st.button('Clear Recommendations'):
        selected_track_id = None
