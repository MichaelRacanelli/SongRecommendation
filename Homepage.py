import streamlit as st
import pandas as pd
from PIL import Image

df = pd.read_csv("spotify_data.csv", index_col=0)

# Add a unique identifier for each song and artist combination
df['identifier'] = df['track_name'] + ' - ' + df['artist_name']

# Sort the DataFrame by popularity
df = df.sort_values(by='popularity', ascending=False)

# Streamlit app title header
st.title('Song Finder')

# Widget for user input (song name)
selected_song = st.text_input('Type in a song name:', value='', key='song_input', help='Make sure to include punctuation!')

# Widget for user input (artist name)
selected_artist = st.text_input('Type in an artist name:', value='', key='artist_input')

# Display information based on user input
if selected_song or selected_artist:
    # Filter DataFrame based on user input
    result = df[
        (df['track_name'].str.contains(selected_song, case=False)) &
        (df['artist_name'].str.contains(selected_artist, case=False))
    ]

    # Subheader for song select
    st.subheader('Select a song:')
    
    # Display the search results as a clickable list with both song and artist names
    selected_identifier = st.selectbox('Select a song by an artist from the results:', 
                                       result['identifier'].tolist(), 
                                       index=0, 
                                       key='selected_song')
    
    # Filter the DataFrame based on the selected identifier
    if selected_identifier:
        selected_song_info = df[df['identifier'] == selected_identifier]
        # Display both song and artist information
        st.write(selected_song_info[['track_name', 'artist_name', 'year', 'genre']].squeeze())

image = Image.open('spotify.jpg')

#Create an about widget
with st.sidebar:
    st.title("We are Group 12 from WatSpeed")
    st.text("And this is our song recommendation app")
    st.image(image, caption='Spotify Genres')
    st.header("Image Credit")
    st.text("Photo by David Pupăză Unsplash")
    
    
  