import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle
from sklearn.naive_bayes import GaussianNB

st.title('Aplikasi Prediksi Gerbong Terbuka UPT Balai Yasa Lahat')
st.write("""
aplikasi berbasis web untuk memprediksi kelayakan gerbong terbuka
Unit Pelaksana Tahunan Balai Yasa Lahat atau UPT. Balai Yasa Lahat adalah tempat untuk melakukan semiperawatan akhir dua tahunan, empat tahunan, serta perbaikan dan kodifikasi sarana perkeretaapian. 
UPT. Balai Yasa Lahat digunakan untuk memperbaiki semua sarana perkeretaapian yang dialokasikan di Divisi Regional III Palembang dan Divisi Regional IV Tanjung Karang.  
""")

# img = Image.open('foto.png')
st.image('foto1.png')

st.sidebar.header('Parameter Inputan')

# upload file CSV untuk parameter inputan
upload_file = st.sidebar.file_uploader("Upload File CSV Anda disini", type=["csv"])
if upload_file is not None:
    inputan = pd.read_csv(upload_file)
    
else:
    def input_user():
        DIPO = st.sidebar.selectbox('DIPO', ('MRL', 'RJS'))
        Boffer_Fixed = st.sidebar.selectbox('Boffer Fixed', ('Baik', 'Rusak'))
        DS = st.sidebar.slider('Dinding Samping (mm)', 10.0, 50.0,30.)
        DU = st.sidebar.slider('Dinding Ujung Diagonal (mm)', 10.0,50.0,30.0)
        TAR = st.sidebar.selectbox('TAR', ('100', '75'))
        data = {'DS': DS,
                'DU': DU,
                'TAR': TAR,
                'Boffer Fixed': Boffer_Fixed,
                'DIPO' : DIPO}

        fitur = pd.DataFrame(data, index=[0])
        return fitur
    inputan = input_user()

# menggabungkan inputan dan dataset gerbong terbuka
gerbong_raw = pd.read_csv('data-gerbongan.csv')
gerbong = gerbong_raw.drop(columns=['hasil'])
df = pd.concat([inputan, gerbong], axis=0)

# encode untuk fitur ordinal 
encode = ['Boffer Fixed','DIPO']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
df = df[:1] #ambil baris pertama(input data user)

# menampilkan parameter hasil inputan
st.subheader('Parameter Inputan')

if upload_file is not None:
    st.write(df)
else:
    st.write('Menunggu file csv untuk di upload. Saat ini memakai sampel inputan (seperti tampilan data frame di bawah ini)')
    st.write(df)


# load model NBC dari file pickle 
load_model = pickle.load(open('dataset.pkl', 'rb'))

# mengambil hanya kolom-kolom yang dibutuhkan untuk prediksi
fitur = ['DS', 'DU', 'TAR', 'Boffer Fixed_Baik', 'Boffer Fixed_Rusak', 'DIPO_MRL', 'DIPO_RJS']
df = df[fitur]

# merombak data frame menjadi array dua dimensi
df = df.to_numpy().reshape(1, -1)

# menerapkan prediksi naive bayes
prediksi = load_model.predict(df)
prediksi_proba = load_model.predict_proba(df)

st.subheader('Keterangan Label Kelas')
hasil = np.array(['layak', 'tidak_layak'])
st.write(hasil)

st.subheader('Hasil Prediksi (Klasifikasi Gerbong Terbuka)')
st.write(hasil[prediksi])

st.subheader('Probabilitas Hasil Prediksi (Klasifikasi Gerbong Terbuka)')
st.write(prediksi_proba) 


