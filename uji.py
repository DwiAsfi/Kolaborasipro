import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# Membaca dataset dari file CSV
df = pd.read_csv('sales.csv')

# Mengubah kolom "Date" menjadi tipe data datetime
df['Date'] = pd.to_datetime(df['Date'])

# Mengurutkan dataset berdasarkan tanggal
df = df.sort_values('Date')

# Memperoleh fitur dan target
X = df.index.values.reshape(-1, 1)
y = df['Sale_volume'].values

# Normalisasi fitur menggunakan MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Memisahkan data menjadi data pelatihan dan data pengujian
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.7, test_size=0.3, shuffle=False)

# Membangun model regresi linear
model = LinearRegression()

# Melatih model menggunakan data pelatihan
model.fit(X_train, y_train)

# Implementasi Streamlit
st.title("Prediksi Sale Volume")
st.write("Masukkan tanggal untuk memprediksi Sale Volume:")

# Input tanggal
input_date = st.date_input("Tanggal")

# Mengonversi input tanggal menjadi indeks numerik
input_index = (pd.to_datetime(input_date) - df['Date'].min()).days

# Menormalisasi input
input_index_scaled = scaler.transform(np.array([[input_index]]))

# Memprediksi Sale Volume
predicted_value = model.predict(input_index_scaled)

st.write("Prediksi Sale Volume:", predicted_value[0])
