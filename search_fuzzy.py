from rapidfuzz import process
import string
from unidecode import unidecode

# import data
# spotify_data = pd.read_csv("spotify_data.csv")

# text cleaning function
def clean_text(user_input):
    string_input = str(user_input)

    # Convert to lowercase
    text = string_input.lower()
    
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
    string_input = "" + user_input
    cleaned_user_input = clean_text(string_input)
    
    # Use rapidfuzz's process.extract function to find the top matches
    # I apply the clean_text function to the spotify data to remove the uppercase letters before comparing
    top_matches = process.extract(cleaned_user_input, data['track_name'].apply(clean_text), limit=top_n)
    
    # Prepare results as a list of dictionaries with track name, artist name, and similarity score
    # we can add more fields as we want
    results = []
    for _, score, index in top_matches:
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
    string_input = "" + user_input
    cleaned_user_input = clean_text(string_input)
    
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


# testing it
# user_input_song = input("Enter a song name: ")

# top_results = find_top_matches(user_input_song, spotify_data)
# top_result = find_top_match(user_input_song, spotify_data)

# # Print the top n results

# for result in top_results:
#     print(f"Match: {result['original_track_name']} | Artist: {result['artist']} | Similarity Score: {result['similarity_score']}")