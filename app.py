import streamlit as st
import pandas as pd 
import numpy as np
import pickle as pkl
import altair as alt
import joblib 

model = joblib.load('music-recommender.joblib')



def main():

    st.set_page_config(page_title='Music Recommender',  layout = 'centered', initial_sidebar_state = 'auto')
    html_temp = """
    <div style="background-color:#f63366 ;padding:10px">
    <h2 style="color:white;text-align:center;">Music Recommender</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    st.text("This application recommends a genre of music based on age and gender")
    st.text("Using Streamlit")

    df = pd.read_csv('music.csv')

    if st.checkbox('Show Dataset'):
        st.header('Dataset')
        st.table(df)

    
    st.header('Add age and gender')
    age = st.slider("Enter your age",20,37,22)
    select_gender = st.selectbox("Select your gender", ("Male","Female"))
    if select_gender=='Male':
        gender = 0
    else:
        gender = 1

    st.success("Predicted Genre of Music: {}".format(model.predict([[age, gender]])))
   
if __name__ == "__main__":
    main()