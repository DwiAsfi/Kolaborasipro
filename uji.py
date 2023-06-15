import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np

# Fungsi tuning model
def tuning(X_train, Y_train, X_test, Y_test, iterasi):
   hasil = 1
   iter = 0
   
   # Normalisasi fitur menggunakan MinMaxScaler
   scaler = MinMaxScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   
   for i in range(1, iterasi):
      model = LinearRegression()
      model = model.fit(X_train_scaled, Y_train)
      y_pred = model.predict(X_test_scaled)
      reshaped_data = y_pred.reshape(-1, 1)
      original_data = scaler.inverse_transform(reshaped_data)
      reshaped_datates = Y_test.reshape(-1, 1)
      actual_test = scaler.inverse_transform(reshaped_datates)
      akhir1 = pd.DataFrame(original_data)
      akhir = pd.DataFrame(actual_test)
      mape = mean_absolute_percentage_error(original_data, actual_test)
      if mape < hasil:
         hasil = mape
         iter = i
   return hasil, iter

# Fungsi untuk melakukan pra-pemrosesan menggunakan Min-Max Scaling
def preprocess_data(data):
   scaler = MinMaxScaler()
   scaled_data = scaler.fit_transform(data)
   return scaled_data

# Fungsi utama aplikasi Streamlit
def main():
   st.title("Prediksi Volume")

   # Membaca dataset dari file CSV
   df = pd.read_csv('mandiri.csv')

   # Mengubah kolom "Date" menjadi tipe data datetime
   df['Date'] = pd.to_datetime(df['Date'])

   # Mengurutkan dataset berdasarkan tanggal
   df = df.sort_values('Date')

   # Memperoleh fitur dan target
   X = df.index.values.reshape(-1, 1)
   y = df['Volume'].values

   # Normalisasi fitur menggunakan MinMaxScaler
   scaler = MinMaxScaler()
   X_scaled = scaler.fit_transform(X)

   # Memisahkan data menjadi data pelatihan dan data pengujian
   X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.7, test_size=0.3, shuffle=False)

   # Membangun model regresi linear
   model = LinearRegression()

   # Melatih model menggunakan data pelatihan
   model.fit(X_train, y_train)

# Implementasi Streamlit menggunakan tab
   tabs = st.tabs(['Data', 'Preprocessing Data', 'Modelling', 'Implementasi'])

   with tabs[0]:
      st.title("""
      Peramalan Data Time Series Pada Saham PT. Bank Mandiri(Persero).
      """)
      st.write('Proyek Sain Data')
      st.text("""
                  Ketua Kelompok :
                  1. Dwi Asfi Fajrin 200411100121

                  Anggota Kelompok :
                  1. Salmatul Farida 200411100016 
                  2. Zakkiya Fitri Nur Sa'adah 200411100097   
                  """)
      st.subheader('Tentang Dataset')
      st.write ("""
      Dataset yang digunakan adalah data time series pada Saham PT. Bank Mandiri(Persero), datanya di dapatkan dari website pada link berikut ini.
      https://finance.yahoo.com/quote/BMRI.JK/history?p=BMRI.JK
      """)
      st.write ("""
         Dataset yang digunakan berjumlah 247 data dan terdapat 7 atribut : 
         """)
      st.write('1. Date : berisi tanggal jalannya perdagangan mulai dari tanggal 15 juni 2022- 15 juni 2023')
      st.write('2. Open : berisi Harga pembukaan pada hari tersebut')
      st.write('3. High : berisi Harga tertinggi pada hari tersebut')
      st.write('4. Low : berisi Harga terendah pada hari tersebut')
      st.write('5. Close : berisi Harga penutup pada hari tersebut')
      st.write('6. Adj. Close : berisi Harga penutupan yang disesuaikan dengan aksi korporasi seperti right issue, stock split atau stock reverset')
      st.write('7. Volume : berisi Volume perdagangan (dalam satuan lembar)')
      st.subheader('Dataset')
      df = pd.read_csv('mandiri.csv')
      df

   with tabs[1]:
      st.subheader('Preprocessing Data')
      st.write("Data Setelah Min-Max Scaling:")
      scaled_df = pd.DataFrame({'Volume': y, 'Scaled_Index': X_scaled.flatten()})
      st.dataframe(scaled_df)

   with tabs[2]:
      st.subheader('Linear Regression')
      st.write("Menggunakan Metode/Model Linear Regression")
      
      # st.write("Masukkan jumlah iterasi:")
      iterasi = 10
      hasil_mape = tuning(X_train, y_train, X_test, y_test, iterasi)
      st.write("Hasil MAPE terbaik:", hasil_mape)
      # st.write("Iterasi terbaik:", iter)

   with tabs[3]:
      st.subheader('Implementasi')
      st.write("Masukkan tanggal untuk memprediksi Volume:")
      input_date = st.date_input("Tanggal")
         
         # Mengubah kolom "Date" menjadi tipe data datetime jika belum
      if not pd.api.types.is_datetime64_any_dtype(df['Date']):
         df['Date'] = pd.to_datetime(df['Date'])

      # Mengonversi input tanggal menjadi tipe data Timestamp
      input_date = pd.to_datetime(input_date)

      # Mengonversi tanggal minimum dalam DataFrame menjadi tipe data Timestamp
      df_min_date = pd.to_datetime(df['Date'].min())

      # Menghitung selisih dalam jumlah hari antara input tanggal dan tanggal minimum
      input_index = (input_date - df_min_date).days

         
         # Menormalisasi input
      input_index_scaled = scaler.transform(np.array([[input_index]]))
         
         # Memprediksi Sale Volume
      predicted_value = model.predict(input_index_scaled)
         
      st.write("Prediksi Volume:", predicted_value[0])
# Menjalankan aplikasi Streamlit
if __name__ == "__main__":
   main()
