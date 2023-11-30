import streamlit as st
import pandas as pd


df = pd.read_csv("spotify_data.csv", index_col=0)

# Add a unique identifier for each song and artist combination
# this is so when the user is selecting a specific song from the list after typing, they have artist info included
df['identifier'] = df['track_name'] + ' - ' + df['artist_name']

# Streamlit app title header
st.title('Song Finder')

# Widget for user input
selected_song = st.text_input('To begin, type in a song name:', value='', key='song_input', help='Make sure to include punctuation!')

# Display information based on user input
if selected_song:
    # Filter DataFrame based on user input
    result = df[df['track_name'].str.contains(selected_song, case=False)]

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
        st.write(selected_song_info[['track_name', 'artist_name','year','genre']].squeeze())