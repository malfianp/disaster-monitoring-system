import streamlit as st
import pandas as pd
import joblib
import random
from keras.models import load_model
from streamlit_extras.app_logo import add_logo
from tools import klasifikasi_gempa, get_info_gempa

# initialize state
if "idx" not in st.session_state:
     st.session_state.idx = random.randint(0,500)

# function
def reset_cb():
        st.session_state.cb1=False
def reset_num_cb():
        st.session_state.cb1=False
        st.session_state.v1=0.00
        st.session_state.v2=0.00
        st.session_state.v3=0.00
        st.session_state.v4=-0.00
        st.session_state.v5=-0.00
        st.session_state.v6=-0.00
def random_num():
        st.session_state.cb1=False
        st.session_state.v1=round(random.uniform(-2.00,2.00),2)
        st.session_state.v2=round(random.uniform(-2.00,2.00),2)
        st.session_state.v3=round(random.uniform(-2.00,2.00),2)
        st.session_state.v4=round(random.uniform(-250.00,250.00),2)
        st.session_state.v5=round(random.uniform(-250.00,250.00),2)
        st.session_state.v6=round(random.uniform(-250.00,250.00),2)
def random_idx():
     st.session_state.idx = random.randint(0,500)
def reset_data_cb():
     st.session_state.cb2 = False

# load scaler & model
scaler_X_gempa = joblib.load('scaler/scaler_X_klasifikasi_gempa.save')
model_klasifikasi_gempa = load_model('model/model_klasifikasi_gempa.h5')
data_gempa = pd.read_csv('dataset/data_simulasi_gempa.csv')

# page config
st.set_page_config(page_title="Earthquake", page_icon="👀")
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
	        header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True) 

# logo
add_logo("image/logo_cat.png", height=80)

# sidebar
with st.sidebar:
    st.info("⬆⬆ Pick a menu above! ⬆⬆")

# title
st.title("Earthquake")

# tabs
tab1_gempa, tab2_gempa= st.tabs(["**Classification (input)**",
                                  "**Classification (dataset)**"])

# tab1
with tab1_gempa:
    # konten
    st.write("**Please input values,**")
    st.button("Click to get random values", on_click=random_num)

    col_value1, col_value2 = st.columns(2)
    with col_value1:
        value1=st.number_input("**aX** (accelerometer-X):", key="v1",
                            min_value=-2.00, 
                            max_value=2.00,
                            # value=0.00,
                            step=0.01,
                            on_change=reset_cb)
        value2=st.number_input("**aY** (accelerometer-Y):", key="v2",
                            min_value=-2.00, 
                            max_value=2.00,
                            # value=0.00,
                            step=0.01,
                            on_change=reset_cb)
        value3=st.number_input("**aZ** (accelerometer-Z):", key="v3",
                            min_value=-2.00, 
                            max_value=2.00,
                            # value=0.00,
                            step=0.01,
                            on_change=reset_cb)
    with col_value2:
        value4=st.number_input("**gX** (gyroscope-X):", key="v4",
                            min_value=-250.00,
                            max_value=250.00,
                            # value=0.00,
                            step=0.01,
                            on_change=reset_cb)
        value5=st.number_input("**gY** (gyroscope-Y):", key="v5",
                            min_value=-250.00,
                            max_value=250.00,
                            # value=0.00,
                            step=0.01,
                            on_change=reset_cb)
        value6=st.number_input("**gZ** (gyroscope-Z):", key="v6",
                            min_value=-250.00,
                            max_value=250.00,
                            # value=0.00,
                            step=0.01,
                            on_change=reset_cb)
    # button 
    col_button11,col_button12= st.columns([1,2])
    confirm_checkbox1 = col_button11.checkbox("Confirm values?", 
                                            key="cb1")

    if confirm_checkbox1:
        col_button12.caption(":green[Confirmed!]")
        df_X = pd.DataFrame([[value1,value2,value3,value4,value5,value6]],
                        columns=['aX','aY','aZ','gX','gY','gZ'])
        
        if col_button11.button("Classify", use_container_width=True,type="primary"):
            col_button12.caption(":green[Classified!]")
            y_pred=klasifikasi_gempa(X=df_X,
                      scaler_X=scaler_X_gempa,
                      model=model_klasifikasi_gempa)
            get_info_gempa(data=y_pred)

            "---" 
            st.button("Click to reset", on_click=reset_num_cb)

# tab2
with tab2_gempa:
    st.write("**Here's a data sample for you,**")
    st.caption(f"index-{st.session_state.idx}")
    data_gempa_sample = data_gempa[st.session_state.idx:st.session_state.idx+1].reset_index(drop=True)

    col_sample_data, col_sample_button = st.columns([3,1])
    col_sample_data.dataframe(data_gempa_sample, use_container_width=True)

    col_sample_button.caption("Resample data?")
    col_sample_button.button("Resample", on_click=random_idx)

    col_button21,col_button22= st.columns([1,2])

    if col_button21.checkbox("Confirm data?", key="cb2"):
        col_button22.caption(":green[Confirmed!]")

        if col_button21.button("Classify ", use_container_width=True,type="primary"):
            col_button22.caption(":green[Classified!]")
            y_pred2=klasifikasi_gempa(X=data_gempa_sample,
                      scaler_X=scaler_X_gempa,
                      model=model_klasifikasi_gempa)
            get_info_gempa(data=y_pred2)

            "---" 
            st.button("Click to reset", on_click=reset_data_cb)

# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# initialize
# if "hold" not in st.session_state:
#      st.session_state.hold = False
# def hold_button():
#     st.session_state.hold = True
# if "rs" not in st.session_state:
#      st.session_state.rs = 0
# def increment_rs(increment=0):
#      st.session_state.rs += increment

    # tab2
# with tab2_gempa:
    # st.write(st.session_state.rs)
    # st.write("ini adalah sample data,")
    # data_gempa_sample1 = data_gempa.sample(1, random_state=st.session_state.rs).reset_index(drop=True)

    # col_sample_data, col_sample_button = st.columns([3,1])

    # col_sample_data.dataframe(data_gempa_sample1, use_container_width=True)
    # col_sample_button.button("sample lain", 
    #                          on_click=increment_rs, 
    #                          kwargs=dict(increment=random.randint(1,3)))

    # if st.checkbox("confirm?",on_change=hold_button):
 
    #      data_gempa_sample1
    #      if st.button("classify"):
    #           data_gempa_sample1