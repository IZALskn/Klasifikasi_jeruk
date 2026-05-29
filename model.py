import streamlit as st 
import pandas as pd 
import joblib

st.set_page_config(
  page_title = 'ML klasifikasi Jeruk',
  page_icon = ':tangerine:'
  )

model = joblib.load('model_klasifikasi_jeruk.joblib')

st.title(':tangerine: belajar klasifikasi jeruk')
st.markdown('aplikasi machine learning untuk memprediksi jeruk')

diameter = st.slider('Diameter', 4.0, 10.0, 6.0)
berat = st.slider('Berat', 75.0, 275.0, 150.0)
tebal_kulit = st.slider('Tebal Kulit', 0.3, 1.6, 0.7)
kadar_gula = st.slider('Kadar Gula', 6.0, 14.0, 10.0)
asal_daerah = st.pills('Asal Daerah', ['Jawa Tengah', 'Kalimantan', 'Jawa Barat'])
warna = st.pills('Warna', ['hijau', 'kuning', 'oranye'])
musim_panen = st.pills('Musim Panen', ['kemarau', 'hujan'])

if st.button('Prediksi', type='primary'):
  data_baru = pd.DataFrame([[diameter, berat, tebal_kulit, kadar_gula, asal_daerah, warna, musim_panen]], columns=['diameter', 'berat', 'tebal_kulit', 'kadar_gula', 'asal_daerah', 'warna', 'musim_panen'])
  prediksi = model.predict(data_baru)[0]
  presentase = max(model.predict_proba(data_baru)[0])
  st.success(f'model memprediksi {prediksi} dengan keyakinan {presentase*100:.2f}%')
  st.balloons()

st.divider()
st.caption('dibuat oleh Izal Rifa i')

