import streamlit as st
from tools import page_config

# page config
page_config(title="Home")

# title
st.title("Disaster Monitoring System")
st.write(f'<hr style="background-color: #E84545; margin-top: 0;'
        'margin-bottom: 0; height: 3px; border: none; border-radius: 3px;">',
        unsafe_allow_html=True,)
st.caption("By: M. Alfian Prasetyo")

# konten
st.write("")
st.subheader("Hi 👋, Welcome to the Disaster Monitoring System App!")
st.write("This application is designed to help you monitor, predict floods, and classify earthquakes.")

col1, col2 = st.columns(2, gap="large")
with col1:
    st.image("image/flood.gif")
    col11,col12,col13=st.columns(3)
    col12.subheader("Flood")
    st.write("<center>Classify flood risk levels and predict water levels for the next 6 hours based on water level and weather data from the past two days</center>",
             unsafe_allow_html=True) 
with col2:
    st.image("image/earthquake.gif")
    col21,col22,col23=st.columns(3)
    col22.subheader("Earthquake")
    st.write("<center>Classify earthquakes based on sensor data (accelerometer and gyroscope)</center>",
             unsafe_allow_html=True)

st.divider()
st.caption("This application aims to provide information and predictions. However, please note that the flood predictions may be less accurate due to limitations in data availability.")
st.caption("Please :red[select a menu on the left] to get started! and follow the instructions provided in each menu to view the results.")
