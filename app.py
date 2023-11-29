import streamlit as st
import pandas as pd

st.write("""
         Hello *world!*
         """)

df = pd.read_csv("spotify_data.csv")
st.write("dataset preview", df.head())