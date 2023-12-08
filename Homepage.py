import streamlit as st
import pandas as pd
from PIL import Image
from search_fuzzy import find_top_matches

image = Image.open('spotify.jpg')

# Create an about widget
with st.sidebar:
    st.title("We are Group 12 from WatSpeed")
    st.text("And this is our song recommendation app")
    st.image(image, caption='Spotify Genres')
    st.header("Image Credit")
    st.text("Photo by David Pupăză Unsplash")

spotify_data = pd.read_csv("spotify_data.csv", index_col=0)

@st.cache_data()
def perform_fuzzy_matching(user_input, data, artist_name=None):
    # Filter data based on the optional artist name
    if artist_name:
        filtered_data = data[data['artist_name'].str.contains(artist_name, case=False, na=False)]
    else:
        filtered_data = data
    
    print("GABRIELA LOGS user_input " + user_input)
    return find_top_matches(user_input, filtered_data)

def main():
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
        top_results = perform_fuzzy_matching(user_input_song, spotify_data, artist_name=user_input_artist)

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

    # Display information about the selected song using extracted song ID, can use selected_track_id for input into ML model or to reference selected song in the dataframe   
    if selected_track_id:
        st.write(f"Perform actions with the selected track ID: {selected_track_id}")
        selected_song_info = spotify_data[spotify_data['track_id'] == selected_track_id]
        # Display both song and artist information
        st.write(selected_song_info[['track_name', 'artist_name', 'year', 'genre']].squeeze())
        
if __name__ == '__main__':
    main()