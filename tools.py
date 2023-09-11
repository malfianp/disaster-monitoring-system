import pandas as pd
import numpy as np
import streamlit as st
from streamlit_extras.app_logo import add_logo

# function
# page config
def page_config(title):
    st.set_page_config(page_title=title, page_icon="👀")
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

# banjir
def klasifikasi_banjir(X,scaler_X,model):
    # scaling
    X_klasifikasi_scaled = scaler_X.transform(X[['height']])
    # predict
    y_klasifikasi = model.predict(X_klasifikasi_scaled,verbose=0)
    y_klasifikasi = np.argmax(y_klasifikasi, axis=1)
    # df
    y_klasifikasi = pd.DataFrame(y_klasifikasi,columns = ['status_pred'])
    df_klasifikasi = X.join(y_klasifikasi)
    return df_klasifikasi

def get_X_prediksi(data, date):
    data_history = data.loc[data['datetime'] <= date].head(400).sort_values(by=['datetime']).reset_index(drop=True) # 288+108=396, 400>396
    data_history['cloudcover_3h (%)'] = data_history['cloudcover (%)'].shift(18)
    data_history['humidity_18h (%)'] = data_history['humidity (%)'].shift(108)
    data_history['height_diff_18h (cm)'] = data_history['height (cm)'] - data_history['height (cm)'].shift(108)
    data_history = data_history.dropna().tail(288).reset_index(drop = True)
    data_history = data_history[['datetime',
                                 'height (cm)',
                                 'windgust (kph)',
                                 'cloudcover_3h (%)',
                                 'humidity_18h (%)',
                                 'height_diff_18h (cm)']]
    return data_history

def prediksi_banjir(data, date, X, scaler_X, scaler_y, model):
    # X
    X_prediksi = X.drop(columns=['datetime']).rename(columns={
                                    'height (cm)':'height',
                                    'windgust (kph)':'windgust',
                                    'cloudcover_3h (%)':'cloudcover_3h',
                                    'humidity_18h (%)':'humidity_18h',
                                    'height_diff_18h (cm)':'height_diff_18h'})
    # scaling
    X_prediksi_scaled = scaler_X.transform(X_prediksi)
    # reshape 
    X_prediksi_scaled = X_prediksi_scaled.reshape(1,288,5)
    # predict
    y_prediksi = model.predict(X_prediksi_scaled,verbose=0)
    # reshape
    y_prediksi = y_prediksi.reshape(36,1)
    # inverse scaling
    y_prediksi_inverse = scaler_y.inverse_transform(y_prediksi)
    y_prediksi_inverse = pd.DataFrame(y_prediksi_inverse, columns = ['height']) # y_pred
    # DATA FUTURE
    data_future = data.loc[data['datetime'] > date].tail(36).sort_values(by=['datetime']).reset_index(drop=True)# 36=step
    data_future = data_future[['datetime','height (cm)']].rename(columns = {'height (cm)':'height_true (cm)'})
    # DF PRED
    if data_future.shape[0] == 36:
        df_pred = data_future.join(y_prediksi_inverse)
    else:
        df_pred = y_prediksi_inverse
    return df_pred

def get_info_banjir(y_klasifikasi, y_pred_status):
    date = y_klasifikasi['date'][0]
    height = y_klasifikasi['height'][0]
    status = y_klasifikasi['status_pred'][0]
    
    # Info klasifikasi
    st.write("**Prediction info 💬**")
    col_title, col_text = st.columns([1,8])

    col_title.write('Datetime')
    col_text.write(f': {str(date)}')

    col_title.write('Height')
    col_text.write(f': {str(height.round(2))} cm')

    siaga0 = ':green[SIAGA 0]'
    siaga1 = ':orange[SIAGA 1]'
    siaga2 = ':red[SIAGA 2]'
    aman = ":green[[AMAN]]"
    waspada = ":orange[[WASPADA]]"
    bahaya = ":red[[BAHAYA]]"

    # kondisi status
    if status == 0:
        col_title.write('Status')
        col_text.write(f': {siaga0}')
        col_title.write("Message")
        if (y_pred_status['status_pred']==0).all():                                                                    # jika semua siaga 0
            col_text.write(f': {aman} Dalam 6 jam kedepan diperkirakan akan tetap berstatus {siaga0}.')
            col_text.write('. Tidak akan terjadi banjir.')
        elif (y_pred_status['status_pred'] == 1).any() and not (y_pred_status['status_pred'] == 2).any():                   # jika ada siaga 1 dan tidak ada siaga 2
            t_siaga1_start = (y_pred_status[y_pred_status['status_pred'] == 1].index.min()+1) * 10
            col_text.write(f': {waspada} Dalam {t_siaga1_start} menit kedepan diperkirakan akan berstatus {siaga1}.')
            col_text.write('. Harap pantau ketinggian air secara berkala.')
        elif (y_pred_status['status_pred'] == 2).any():                                                                # jika ada siaga 2
            t_siaga2_start = (y_pred_status[y_pred_status['status_pred'] == 2].index.min()+1) * 10
            col_text.write(f': {bahaya} Dalam {t_siaga2_start} menit kedepan diperkirakan akan berstatus {siaga2}.')
            col_text.write('. Berkemungkinan terjadi banjir, segera lakukan evakuasi.')
        else: col_text.write(': -')
    
    elif status == 1:
        col_title.write('Status')
        col_text.write(f': {siaga1}')
        col_title.write("Message")
        if (y_pred_status['status_pred']==0).all():
            col_text.write(f': {aman} Dalam 10 menit kedepan diperkirakan akan berstatus {siaga0}.')
            col_text.write('. Tidak akan terjadi banjir.')
        elif (y_pred_status['status_pred']==0).any() and not (y_pred_status['status_pred'] == 2).any():
            t_siaga1_end = (y_pred_status[y_pred_status['status_pred'] == 1].index.max()+2) * 10
            col_text.write(f': {aman} Dalam {t_siaga1_end} menit kedepan diperkirakan akan berstatus {siaga0}.')
            col_text.write('. Tidak akan terjadi banjir.')
        elif (y_pred_status['status_pred']==1).all():
            col_text.write(f': {waspada} Dalam 6 jam kedepan diperkirakan akan tetap berstatus {siaga1}.')
            col_text.write('. Harap pantau ketinggian air secara berkala.')
        elif (y_pred_status['status_pred']==2).any():
            t_siaga2_start = (y_pred_status[y_pred_status['status_pred'] == 2].index.min()+1) * 10
            col_text.write(f': {bahaya} Dalam {t_siaga2_start} menit kedepan diperkirakan akan berstatus {siaga2}.')
            col_text.write('. Berkemungkinan terjadi banjir, segera lakukan evakuasi.')
        else: col_text.write(': -')
        
    elif status == 2:
        col_title.write('Status')
        col_text.write(f': {siaga2}')
        col_title.write("Message")
        if not (y_pred_status['status_pred']==2).any():
            col_text.write(f': {waspada} Dalam 10 menit kedepan diperkirakan status {siaga2} akan berakhir.')
            col_text.write('. Harap pantau ketinggian air secara berkala.')
        elif (y_pred_status['status_pred']==2).any() and not (y_pred_status['status_pred']==2).all():
            t_siaga2_end = (y_pred_status[y_pred_status['status_pred'] == 2].index.max()+2) * 10
            col_text.write(f': {bahaya} Dalam {t_siaga2_end} menit kedepan diperkirakan masih berstatus {siaga2}.')
            col_text.write('. Berkemungkinan terjadi banjir, segera lakukan evakuasi.')
        elif (y_pred_status['status_pred']==2).all():
            col_text.write(f': {bahaya} Dalam 6 jam kedepan diperkirakan akan tetap berstatus {siaga2}.')
            col_text.write('. Berkemungkinan terjadi banjir, segera lakukan evakuasi.')
        else: col_text.write(': -')

# gempa
def klasifikasi_gempa(X, scaler_X, model):
    X_data=X.iloc[:,:6]
    X_scaled = scaler_X.transform(X_data)
    X_scaled=X_scaled.reshape(1,1,6)
    # predict
    y_pred = model.predict(X_scaled, verbose=0)
    y_pred = np.argmax(y_pred, axis=1)
    # df
    df_y_pred = pd.DataFrame(y_pred, columns=['result_pred'])
    df_klasifikasi = X.join(df_y_pred)
    return df_klasifikasi
def get_info_gempa(data):
    data = data.rename(columns = {'aX':'aX (g)',
                                  'aY':'aY (g)',
                                  'aZ':'aZ (g)',
                                  'gX':'gX (deg/s)',
                                  'gY':'gY (deg/s)',
                                  'gZ':'gZ (deg/s)'})
    status_pred = data['result_pred'][0]
    if status_pred == 1:
        st.error("Result: Earthquake")
    else:
        st.success("Result: Non-Earthquake")
    with st.expander("Explore classified data"):
                col_ex1,col_ex2 = st.columns([3,1])
                col_ex1.dataframe(data)
                col_ex2.markdown(":green[0 = Non-Earthquake]")
                col_ex2.markdown(":red[1 = Earthquake]")
