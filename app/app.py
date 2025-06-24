# =============================================
# ✅ Streamlit Dashboard — Simple Real-time 
# Model RF + Simulasi Pembelian ON THE FLY
# =============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ---------------------------------------------
# ✅ Load Model Global
# ---------------------------------------------
model = joblib.load('../model/model_global_rf.pkl')

# ---------------------------------------------
# ✅ Config Page
# ---------------------------------------------
st.set_page_config(page_title="Dashboard H- Lebaran — Real-time 🚍", layout="wide")
st.title("📊 Dashboard H- Lebaran (Live Model + Simulasi)")

# ---------------------------------------------
# ✅ Sidebar — Input
# ---------------------------------------------
st.sidebar.header("🔍 Parameter")

tahun = st.sidebar.selectbox("Tahun Lebaran", [2026])
trayek = st.sidebar.selectbox("Trayek", ["TrayekA", "TrayekB", "TrayekC"])

# Range H_minus 
H_min = st.sidebar.slider("H_minus Minimum", -30, -1, -10)
H_max = st.sidebar.slider("H_minus Maximum", 0, 10, 5)

# Lead_time distribusi simulasi
mean_lead = st.sidebar.slider("Mean Lead Time (days)", 10, 60, 30)
std_lead = st.sidebar.slider("Std Dev Lead Time", 1, 10, 3)

# ---------------------------------------------
# ✅ 1) Prediksi Keberangkatan (H_min to H_max)
# ---------------------------------------------
H_range = np.arange(H_min, H_max + 1)
X_pred = pd.DataFrame({
    'H_minus': H_range,
    'Tahun': tahun
})
Y_pred = model.predict(X_pred)

df_keberangkatan = pd.DataFrame({
    'H_minus': H_range,
    'Pred_Penjualan': Y_pred[:, 0],
    'Pred_Keberangkatan': Y_pred[:, 1]
})

# ---------------------------------------------
# ✅ 2) Simulasi Pembelian (ON THE FLY)
# ---------------------------------------------

# Buat lead_time distribusi (Normal)
np.random.seed(42)
lead_time_dist = np.abs(np.random.normal(loc=mean_lead, scale=std_lead, size=1000))
lead_time_dist = np.round(lead_time_dist)
lead_time_dist = lead_time_dist[lead_time_dist > 0]
lead_time_df = pd.Series(lead_time_dist).value_counts(normalize=True).reset_index()
lead_time_df.columns = ['lead_time', 'proporsi']

# Simulasi pembelian dari prediksi keberangkatan
simulasi_rows = []
for _, row in df_keberangkatan.iterrows():
    tgl_keberangkatan = pd.Timestamp(f"{tahun}-03-20") - pd.to_timedelta(row['H_minus'], unit='D')
    total = row['Pred_Keberangkatan']
    for _, lt_row in lead_time_df.iterrows():
        lt = int(lt_row['lead_time'])
        proporsi = lt_row['proporsi']
        tgl_beli = tgl_keberangkatan - pd.Timedelta(days=lt)
        estimasi_beli = total * proporsi
        simulasi_rows.append({
            'Tanggal_Beli': tgl_beli,
            'Estimasi_Pembelian': estimasi_beli
        })

df_pembelian = pd.DataFrame(simulasi_rows)
df_pembelian = df_pembelian.groupby('Tanggal_Beli')['Estimasi_Pembelian'].sum().reset_index()

# ---------------------------------------------
# ✅ 3) Plot Timeline
# ---------------------------------------------
st.subheader(f"🎯 Timeline Trayek: {trayek}")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df_keberangkatan['H_minus'], df_keberangkatan['Pred_Keberangkatan'], label="Keberangkatan", marker='o')
ax.set_xlabel("H_minus")
ax.set_ylabel("Keberangkatan")
ax.set_title("Prediksi Keberangkatan")
ax.grid(True)
ax.legend()
st.pyplot(fig)

fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(df_pembelian['Tanggal_Beli'], df_pembelian['Estimasi_Pembelian'], label="Pembelian", marker='x')
ax2.set_xlabel("Tanggal Beli")
ax2.set_ylabel("Estimasi Pembelian")
ax2.set_title("Simulasi Pembelian")
ax2.grid(True)
ax2.legend()
st.pyplot(fig2)

# ---------------------------------------------
# ✅ 4) Tabel Data
# ---------------------------------------------
st.subheader("📑 Tabel Keberangkatan")
st.dataframe(df_keberangkatan)

st.subheader("📑 Tabel Pembelian (Simulasi)")
st.dataframe(df_pembelian)

st.success("✅ Done! Real-time prediksi & simulasi berjalan mulus 🚀")
