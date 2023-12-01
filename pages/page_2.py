import streamlit as st

st.title("This is where our code will go")

code = '''def hello():
    print("Hello, Streamlit!")'''
st.code(code, language='python')