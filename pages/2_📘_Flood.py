import streamlit as st
import pandas as pd
import pickle
import datetime
import joblib
from keras.models import load_model
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.stylable_container import stylable_container
from tools import *

# page config
page_config(title="Flood")

# function
def plot_tab1(range, data_plot):
    st.caption(f"Here's the height chart for {range}")
    st.area_chart(data=data_plot, 
                      y="height (cm)")
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
def data_plot_tab2(history, pred):
    # history
    data_history = history.reset_index()
    data_history['index'] = data_history['index'] - 287
    data_history = data_history.set_index('index')
    data_plot_X = data_history[['height (cm)']]
    data_plot_X = data_plot_X.rename(columns = {'height (cm)':'height_history (cm)'})
    # future
    data_pred = pred.reset_index()
    data_pred['index'] = data_pred['index'] + 1
    data_pred = data_pred.set_index('index')
    #concat
    data_plot_2 = pd.concat([data_plot_X, data_pred])
    return data_history, data_pred, data_plot_2

@st.cache_resource(ttl=3600)
def load_data_banjir():
     with open("dataset/data_simulasi_banjir.pkl", 'rb') as file:
         data = pickle.load(file)
     return data

@st.cache_resource(ttl=3600)
def load_scaler_banjir():
     scaler_X_klas = joblib.load('scaler/scaler_X_klasifikasi_banjir.save')
     scaler_X_pred = joblib.load('scaler/scaler_X_prediksi_banjir.save') 
     scaler_y_pred = joblib.load('scaler/scaler_y_prediksi_banjir.save') 
     return scaler_X_klas, scaler_X_pred, scaler_y_pred

@st.cache_resource(ttl=3600)
def load_model_banjir():
     model_klas = load_model('model/model_klasifikasi_banjir.h5')
     model_pred = load_model('model/model_prediksi_banjir.h5')
     return model_klas, model_pred

# load data, scaler, model
data_banjir = load_data_banjir()
scaler_X_klasifikasi, scaler_X_prediksi, scaler_y_prediksi = load_scaler_banjir()
model_klasifikasi_banjir, model_prediksi_banjir = load_model_banjir()

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
    data_tab1 = data_banjir.loc[data_banjir['datetime']<=datetime1].reset_index(drop=True)
    data_metric = data_tab1[['datetime','height (cm)']].head(2)
    data_metric = data_metric.rename(columns={'datetime':'date',
                                             'height (cm)':'height'})
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
    data_plot1 = data_tab1[['datetime','height (cm)']].set_index("datetime")        # set 'date' as index
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
        X_klasifikasi = data_banjir.loc[data_banjir['datetime'] == datetime2].reset_index(drop=True)
        X_klasifikasi = X_klasifikasi[['datetime','height (cm)']].rename(columns={'datetime':'date',
                                                                                  'height (cm)':'height'})
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
        y_pred_status = y_pred_status.rename(columns = {'height':'height_pred (cm)'})

        # info
        with stylable_container(key="container_with_border",
                                css_styles="""{
                                border: 3px solid #dedede;
                                border-radius: 0.5rem;
                                padding: calc(1em - 1px)}"""):
            get_info_banjir(y_klasifikasi=y_klasifikasi,
                             y_pred_status=y_pred_status)

        # plot grafik    
        st.write("**Plot prediction data 📈**")

        if y_pred_status.shape[1] == 4:
            data_historical,data_future,data_plot2 = data_plot_tab2(history=X_prediksi, 
                                                                    pred=y_pred_status)
            st.area_chart(data=data_plot2,
                      y=["height_history (cm)", 
                         "height_true (cm)", 
                         "height_pred (cm)"])
            
        elif y_pred_status.shape[1] == 2:
            data_historical,data_future,data_plot2 = data_plot_tab2(history=X_prediksi, 
                                                                    pred=y_pred_status)
            st.area_chart(data=data_plot2,
                      y=["height_history (cm)",
                         "height_pred (cm)"])
        
        # data
        with st.expander("**Explore prediction data**"):
            st.write("**Historical data:**")
            st.dataframe(data_historical)
            st.write("**Future/predicted data:**")
            st.dataframe(data_future)

        "---"      
        st.button("Click to reset")
