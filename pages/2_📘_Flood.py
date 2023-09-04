import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
import joblib
from keras.models import load_model
from streamlit_extras.app_logo import add_logo
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.stylable_container import stylable_container
from tools import klasifikasi_banjir, get_X_prediksi, prediksi_banjir, get_info_banjir

# function
def plot_tab1(range, data_plot):
    st.caption(f"Here's the height chart for {range}")
    st.area_chart(data=data_plot, 
                      y="height")
def convert_df(df):
        return df.to_csv().encode('utf-8')
def download_csv(data_df, name):
    st.dataframe(data_df)
    st.caption("Need the data?")
    if st.download_button(label="Download data as CSV",
                           data=convert_df(data_df),
                           file_name=name,
                           mime='text/csv'):
        st.toast('Download complete!', icon = "✅")

# load data, scaler, model
with open("dataset/data_simulasi_banjir.pkl", 'rb') as file:
    data_banjir = pickle.load(file)
scaler_X_klasifikasi = joblib.load('scaler/scaler_X_klasifikasi_banjir.save')
scaler_X_prediksi = joblib.load('scaler/scaler_X_prediksi_banjir.save') 
scaler_y_prediksi = joblib.load('scaler/scaler_y_prediksi_banjir.save') 
model_klasifikasi_banjir = load_model('model/model_klasifikasi_banjir.h5')
model_prediksi_banjir = load_model('model/model_prediksi_banjir.h5')

# page config
st.set_page_config(page_title="Banjir", page_icon="👀")
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
st.title("Flood")
# tabs
tab1, tab2 = st.tabs(["**Dashboard**", "**Prediction**"])

# konten tab 1 (Dashboard)
with tab1:
    col_datetime1, col_metric1, col_metric2= st.columns(3)
    # datetime
    with col_datetime1:
        date1 = st.date_input("d", 
                               value=datetime.datetime(2022,9,30),
                               min_value=datetime.datetime(2022,9,9),
                               max_value=datetime.datetime(2022,9,30),
                               label_visibility="collapsed")
        time1 = st.time_input("t",
                               value=datetime.time(10,30),
                               step=600,
                               label_visibility="collapsed")
        st.caption("Set date and time 👆")
    # data tab 1
    datetime1 = str(date1)+" "+str(time1)
    data_tab1 = data_banjir.loc[data_banjir['date']<=datetime1].reset_index(drop=True)
    data_metric = data_tab1.head(2)
    data_metric = klasifikasi_banjir(X=data_metric,        # get status
                                     scaler_X=scaler_X_klasifikasi,
                                     model=model_klasifikasi_banjir)
    
    metric1_value = data_metric.height[0].round(2)
    metric1_delta = (data_metric.height[0] - data_metric.height[1]).round(2)
    
    metric2_value = data_metric.status_pred[0]
    metric2_delta = int(metric2_value - data_metric.status_pred[1])

    # metric 1
    with col_metric1:
        st.metric(label="Last height:",
                  value=f"{metric1_value} cm",
                  delta=metric1_delta,
                  delta_color="inverse")
    # metric 2
    with col_metric2:
        st.metric(label="Last status:",
                  value=f"Siaga {metric2_value}",
                  delta=metric2_delta,
                  delta_color="off")
    # metric style
    style_metric_cards(background_color="#f0f2f6",
                       border_left_color="#E84545",
                       border_size_px=0, 
                       border_radius_px=10,
                       box_shadow=False)

    # plot
    data_plot1 = data_tab1.set_index("date")        # set 'date' as index
    col_plot1,col_plot2 = st.columns([2,1])
    col_plot1.subheader("Visualizing height changes:")
    data_plot1_range = col_plot2.selectbox("Plot range:", 
                        ("last day","last week","last month","last year","all data"),
                        label_visibility="collapsed")
    if data_plot1_range == "last day":                     # last day
        plot_tab1(range="the last day",
                  data_plot=data_plot1.head(144))
    elif data_plot1_range == "last week":                  # last week
        plot_tab1(range="the last week",
                  data_plot=data_plot1.head(1008))
    elif data_plot1_range == "last month":                 # last month
        plot_tab1(range="the last month",
                  data_plot=data_plot1.head(4320))
    elif data_plot1_range == "last year":                  # last year
        plot_tab1(range="the last year",
                  data_plot=data_plot1.head(52560))     
    elif data_plot1_range == "all data":                   # all data
        plot_tab1(range="all data",
                  data_plot=data_plot1)

    # data history
    col_data1,col_data2 = st.columns([2,1])
    col_data1.subheader("Historical data table:")
    data_tab1_range = col_data2.selectbox("Data range:", 
                        ("last day","last week","last month","last year","all data"),
                        label_visibility="collapsed")
    if data_tab1_range == "last day":                      # last day
        download_csv(data_df=data_tab1.head(144),
                     name="last_day.csv")
    elif data_tab1_range == "last week":                   # last week
        download_csv(data_df=data_tab1.head(1008),
                     name="last_week.csv")
    elif data_tab1_range == "last month":                  # last month
        download_csv(data_df=data_tab1.head(4320),
                     name="last_month.csv")
    elif data_tab1_range == "last year":                   # last year
        download_csv(data_df=data_tab1.head(52560),
                     name="last_year.csv")
    elif data_tab1_range == "all data":                    # all data
        download_csv(data_df=data_tab1,
                     name="all_data.csv")

with tab2:
    col_date2, col_time2, col_caption2= st.columns([1,1,1])
    # datetime
    date2 = col_date2.date_input("Date:",
                                 value=datetime.datetime(2022,9,30),
                                 min_value=datetime.datetime(2022,9,11),
                                 max_value=datetime.datetime(2022,9,30),
                                 label_visibility="collapsed")
    time2 = col_time2.time_input("Time:",
                                 value=datetime.time(4,30),
                                 step=600,
                                 label_visibility="collapsed")
    col_caption2.caption("👈 Set date and time")
    datetime2 = str(date2)+" "+str(time2)
  
    # proses
    col_button1,col_button2= st.columns([1,2])
    if col_button1.button("Predict ", use_container_width=True,type="primary"):
        col_button2.caption(":green[Predicted!]")
        # klasifikasi
        X_klasifikasi = data_banjir.loc[data_banjir['date'] == datetime2].reset_index(drop=True)
        X_klasifikasi = X_klasifikasi[['date','height']]  
        y_klasifikasi = klasifikasi_banjir(X=X_klasifikasi,
                                 scaler_X=scaler_X_klasifikasi,
                                 model=model_klasifikasi_banjir)            
        # prediksi
        X_prediksi = get_X_prediksi(data=data_banjir, 
                                date=datetime2)
        y_pred = prediksi_banjir(data=data_banjir,
                            date=datetime2,
                            X=X_prediksi,
                            scaler_X=scaler_X_prediksi,
                            scaler_y=scaler_y_prediksi,
                            model=model_prediksi_banjir)
        y_pred_status = klasifikasi_banjir(X=y_pred,
                                    scaler_X=scaler_X_klasifikasi,
                                    model=model_klasifikasi_banjir)
        y_pred_status = y_pred_status.rename(columns = {'height':'height_pred'})

        with stylable_container(key="container_with_border",
                                css_styles="""{
                                border: 3px solid #dedede;
                                border-radius: 0.5rem;
                                padding: calc(1em - 1px)}"""):
             
            get_info_banjir(y_klasifikasi=y_klasifikasi,
                             X=X_prediksi,
                             y_pred_status=y_pred_status)
            
        # plot
        X_prediksi = X_prediksi.reset_index()
        X_prediksi['index'] = X_prediksi['index'] - 287
        X_prediksi = X_prediksi.set_index('index')
        data_plot_X = X_prediksi[["height"]]
        data_plot_X = data_plot_X.rename(columns = {'height':'height_history'})

        st.write("**Plot prediction data 📈**")
        
        if y_pred_status.shape[1] == 4:
            data_plot_y_pred = y_pred_status.reset_index()
            data_plot_y_pred['index'] = data_plot_y_pred['index'] + 1
            data_plot_y_pred = data_plot_y_pred.set_index('index')
            data_plot2 = pd.concat([data_plot_X, data_plot_y_pred])
            st.area_chart(data=data_plot2,
                      y=["height_history", "height_true", "height_pred"])
            
        elif y_pred_status.shape[1] == 2:
            data_plot_y_pred = y_pred_status.reset_index()
            data_plot_y_pred['index'] = data_plot_y_pred['index'] + 1
            data_plot_y_pred = data_plot_y_pred.set_index('index')
            data_plot2 = pd.concat([data_plot_X, data_plot_y_pred])
            st.area_chart(data=data_plot2,
                      y=["height_history",  "height_pred"])
            
        with st.expander("**Explore prediction data**"):
            st.write("**Historical data:**")
            st.dataframe(X_prediksi)
            st.write("**Future/predicted data:**")
            st.dataframe(data_plot_y_pred)

        "---"      
        st.button("Click to reset")
