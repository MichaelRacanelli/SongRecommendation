import streamlit as st
import pandas as pd
from PIL import Image
from rapidfuzz import process
import string
from unidecode import unidecode

spotify_data = pd.read_csv("spotify_data.csv", index_col=0)

# Add a unique identifier for each song and artist combination
# df['identifier'] = df['track_name'] + ' - ' + df['artist_name']

# Streamlit app title header
st.title('Song Finder')

# text cleaning function
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Convert special characters to closest ASCII equivalents
    text = unidecode(text)
    
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Remove double whitespaces
    text = ' '.join(text.split())
    
    # Remove leading and trailing whitespaces
    text = text.strip()
    
    return text


# fuzzy matching function
# Returns n top matches

# (uses rapidfuzz, which uses a variation of the Levenshtein distance)
# https://maxbachmann.github.io/RapidFuzz/Usage/fuzz.html

def find_top_matches(user_input, data, top_n=10):
    # clean user_input
    cleaned_user_input = clean_text(user_input)
    
    # Use rapidfuzz's process.extract function to find the top matches
    # I apply the clean_text function to the spotify data to remove the uppercase letters before comparing
    top_matches = process.extract(cleaned_user_input, data['track_name'].apply(clean_text), limit=top_n)
    
    # Prepare results as a list of dictionaries with track name, artist name, and similarity score
    # we can add more fields as we want
    results = []
    for match, score, index in top_matches:
        original_track_name = data.loc[index, 'track_name']
        artist_name = data.loc[index, 'artist_name']
        track_id = data.loc[index, 'track_id']
        result_dict = {
            'track_id': track_id,
            'original_track_name': original_track_name,
            'artist': artist_name,
            'similarity_score': score
        }
        results.append(result_dict)
    
    return results


# same function, but returns only the top match
def find_top_match(user_input, data):
    # clean user_input
    cleaned_user_input = clean_text(user_input)
    
    # Use rapidfuzz's process.extract function to find the top matches
    # I apply the clean_text function to the spotify data to remove the uppercase letters before comparing
    top_match, score, index = process.extractOne(cleaned_user_input, data['track_name'].apply(clean_text))
    
    # Prepare the result as a dictionary
    result = {
        'original_track_name': data.loc[index, 'track_name'],
        'artist': data.loc[index, 'artist_name'],
        'similarity_score': score
    }
    
    return result

# Widget for user input (song name)
user_input_song = st.text_input('Type in a song name:', value='', key='song_input', help='Type in the song you are looking for!')

# Widget for user input (artist name)
# selected_artist = st.text_input('Type in an artist name:', value='', key='artist_input')

# Placeholder for selected track ID
selected_track_id = None

#actions to be performed when the user inputs a song in the search box
if user_input_song:
    #generate fuzzy matching top results for search input
    top_results = find_top_matches(user_input_song, spotify_data)
    # top_result = find_top_match(user_input_song, spotify_data)

    #create a list item of clean song, artist, and similarity score for displaying in the selectbox widget
    clean_results = []
    for result in top_results:
        clean_results.append(f"{result['original_track_name']} by {result['artist']} | Similarity Score: {result['similarity_score']}")

    # Display the clean search results as a clickable list with both song and artist names with similarity score
    st.subheader('Select a song:')
    selected_song_index = st.selectbox('Select a song by an artist from the results:', 
                                       range(len(clean_results)),  # Use range(len(clean_results)) as the options
                                       format_func=lambda x: clean_results[x],  # Display clean_results in the dropdown
                                       index=0, 
                                       key='selected_song')
    
    # Extract the track ID from the selected result
    if selected_song_index is not None:
        selected_track_id = top_results[selected_song_index]['track_id']

#display information about the selected song using extracted song ID, can use selected_track_id for input into ML model or to reference selected song in the dataframe   
if selected_track_id:
    st.write(f"Perform actions with the selected track ID: {selected_track_id}")
    selected_song_info = spotify_data[spotify_data['track_id'] == selected_track_id]
    # Display both song and artist information
    st.write(selected_song_info[['track_name', 'artist_name', 'year', 'genre']].squeeze())
    
#old code if needed for artist search
# # Display information based on user input
# if selected_song or selected_artist:
#     # Filter DataFrame based on user input
#     result = df[
#         (df['track_name'].str.contains(selected_song, case=False)) &
#         (df['artist_name'].str.contains(selected_artist, case=False))
#     ]

#     # Subheader for song select
#     st.subheader('Select a song:')
    
#     # Display the search results as a clickable list with both song and artist names
#     selected_identifier = st.selectbox('Select a song by an artist from the results:', 
#                                        result['identifier'].tolist(), 
#                                        index=0, 
#                                        key='selected_song')
    
#     # Filter the DataFrame based on the selected identifier
#     if selected_identifier:
#         selected_song_info = df[df['identifier'] == selected_identifier]
#         # Display both song and artist information
#         st.write(selected_song_info[['track_name', 'artist_name', 'year', 'genre']].squeeze())

image = Image.open('spotify.jpg')

#Create an about widget
with st.sidebar:
    st.title("We are Group 12 from WatSpeed")
    st.text("And this is our song recommendation app")
    st.image(image, caption='Spotify Genres')
    st.header("Image Credit")
    st.text("Photo by David Pupăză Unsplash")
    
    
  