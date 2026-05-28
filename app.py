import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------- Konfigurasi Halaman ----------
st.set_page_config(
    page_title='Prediksi Harga Rumah - Depok',
    page_icon='🏢',
    layout='wide',
)

# ---------- Super Premium CSS Injection (SaaS Theme) ----------
st.markdown("""
    <style>
    /* Import Font Modern dari Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    /* Terapkan Font ke Seluruh Elemen Aplikasi */
    html, body, [class*="css"], .stMarkdown, label {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* 1. Mengubah Background Utama Aplikasi */
    [data-testid="stAppViewContainer"] {
        background-color: #F8FAFC;
    }
    
    
    /* 2. Mengubah Sidebar Menjadi Dark Charcoal Premium */
    [data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #E2E8F0;
    }
    
    /* Styling Komponen di Dalam Sidebar */
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label {
        color: #F1F5F9 !important;
        font-weight: 600 !important;
    }
    
    /* 3. Judul Utama dengan Efek Multi-Gradient Premium */
    .main-title {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 50%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }
    
    .sub-title {
        font-size: 16px;
        color: #64748B;
        margin-bottom: 35px;
        font-weight: 500;
    }
    
    /* 4. Desain Kartu Konten */
    .premium-card {
        background-color: #FFFFFF;
        padding: 28px;
        border-radius: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.03), 0 8px 10px -6px rgba(0, 0, 0, 0.03);
        border: 1px solid #E2E8F0;
        margin-bottom: 25px;
    }
    
    /* 5. HERO BANNER HASIL PREDIKSI */
    .hero-prediction-box {
        background: linear-gradient(135deg, #1E3A8A 0%, #1D4ED8 100%);
        color: white;
        padding: 35px;
        border-radius: 24px;
        box-shadow: 0 20px 25px -5px rgba(29, 78, 216, 0.15);
        text-align: center;
        margin-bottom: 30px;
    }
    
    .hero-price {
        font-size: 48px;
        font-weight: 800;
        color: #FFFFFF;
        margin-top: 10px;
        letter-spacing: -0.5px;
        text-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* 6. Desain Kartu Selamat Datang */
    .welcome-card {
        background: #FFFFFF;
        padding: 40px;
        border-radius: 24px;
        border: 1px solid #E2E8F0;
        border-top: 8px solid #2563EB;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.05);
        text-align: center;
        max-width: 800px;
        margin: 40px auto;
    }
    
    /* 7. Kustomisasi Tombol Utama */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        padding: 14px 24px !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3) !important;
    }
    
    div.stButton > button:first-child:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 25px rgba(37, 99, 235, 0.5) !important;
        background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%) !important;
    }
    
    /* 8. Menghias Tampilan Badge Informasi Grid */
    .grid-badge {
        background-color: #F8FAFC;
        padding: 15px;
        border-radius: 12px;
        border-left: 4px solid #3B82F6;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Muat Artefak & Data Mapping ----------
@st.cache_resource
def load_artefak():
    model  = joblib.load('regresi_berganda.pkl')
    scaler = joblib.load('scaler.pkl')
    fitur  = joblib.load('fitur.pkl')
    return model, scaler, fitur

@st.cache_data
def load_mapping_lokasi():
    df_clean = pd.read_csv('data_rumah_clean.csv')
    mapping = df_clean.groupby('Lokasi')['Lokasi_encoded'].first().to_dict()
    return mapping

model, scaler, FITUR = load_artefak()
mapping_lokasi = load_mapping_lokasi()

# ---------- Header Utama Aplikasi (SUDAH DIPERBAIKI) ----------
st.markdown('<div style="font-size: 42px; font-weight: 800; margin-bottom: 5px;">🏢 <span class="main-title">Aplikasi Prediksi Harga Rumah Wilayah Depok</span></div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Sistem Penaksiran Nilai Jual Properti Berbasis Machine Learning Multiple Regression OLS</div>', unsafe_allow_html=True)

# ---------- Pembuatan Menu Tab ----------
tab1, tab2 = st.tabs(["🔮 Engine Kalkulator", "📊 Transparansi Algoritma"])

# ---------- Input Skenario di Sidebar ----------
st.sidebar.markdown("<h2 style='color: #F8FAFC; font-size: 22px; font-weight:700; margin-bottom: 20px;'>⚙️ Spesifikasi Properti</h2>", unsafe_allow_html=True)
input_user = {}

for f in FITUR:
    if f == 'Lokasi_encoded':
        lokasi_terpilih = st.sidebar.selectbox(
            label='📍 Wilayah Kecamatan',
            options=sorted(list(mapping_lokasi.keys())),
            help='Pilih lokasi kecamatan properti di wilayah Depok.'
        )
        input_user[f] = mapping_lokasi[lokasi_terpilih]
        
    elif f in ['Luas Tanah', 'Luas Bangunan']:
        input_user[f] = st.sidebar.number_input(
            label=f'📐 {f} (m²)',
            min_value=10,
            value=100,
            step=5,
            format='%d'
        )
        
    elif f in ['Kamar Tidur', 'Kamar Mandi', 'Garasi']:
        emoji = '🛏️' if f == 'Kamar Tidur' else ('🚿' if f == 'Kamar Mandi' else '🚗')
        input_user[f] = st.sidebar.number_input(
            label=f'{emoji} Total {f}',
            min_value=0,
            value=2,
            step=1,
            format='%d'
        )

st.sidebar.markdown("<br>", unsafe_allow_html=True)
tombol_prediksi = st.sidebar.button('🚀 Jalankan Analisis Harga', use_container_width=True)

# ---------- TAB 1: ENGINE KALKULATOR PREDIKSI ----------
with tab1:
    if tombol_prediksi:
        try:
            nilai = pd.DataFrame([[input_user[f] for f in FITUR]], columns=FITUR)
            nilai_sc = scaler.transform(nilai)
            pred = model.predict(nilai_sc)[0]

            if pred < 0:
                st.error('⚠️ Hasil perhitungan matematis menghasilkan nilai negatif. Silakan tinjau kembali input karakteristik rumah.')
            else:
                st.markdown(f"""
                <div class="hero-prediction-box">
                    <div style="font-size: 16px; font-weight: 600; text-transform: uppercase; letter-spacing: 2px; color: #93C5FD;">
                        Hasil Estimasi Nilai Pasar Wajar
                    </div>
                    <div class="hero-price">Rp {pred:,.0f}</div>
                    <div style="font-size: 14px; color: #BFDBFE; margin-top: 10px; font-style: italic;">
                        *Dihitung secara real-time berdasarkan matriks korelasi data properti Depok
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('<div class="premium-card">', unsafe_allow_html=True)
                st.markdown("<h3 style='margin-top:0px; font-size:20px; color:#1E3A8A; font-weight:700;'>📋 Ringkasan Atribut Rumah Jual</h3>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"<div class='grid-badge'><b>📍 Wilayah Kecamatan:</b><br>{lokasi_terpilih}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='grid-badge'><b>📐 Luas Bidang Tanah:</b><br>{input_user['Luas Tanah']} m²</div>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<div class='grid-badge'><b>🛏️ Kamar Tidur Utama:</b><br>{input_user['Kamar Tidur']} Ruang</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='grid-badge'><b>📐 Luas Fisik Bangunan:</b><br>{input_user['Luas Bangunan']} m²</div>", unsafe_allow_html=True)
                with col3:
                    st.markdown(f"<div class='grid-badge'><b>🚿 Kamar Mandi Bersih:</b><br>{input_user['Kamar Mandi']} Ruang</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='grid-badge'><b>🚗 Kapasitas Area Garasi:</b><br>{input_user['Garasi']} Kendaraan</div>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f'Terjadi gangguan teknis sistem: {e}')
    else:
        st.markdown(f"""
        <div class="welcome-card">
            <div style="font-size: 50px; margin-bottom: 15px;">📊</div>
            <h3 style="color: #0F172A; font-weight: 800; margin-top: 0px;">Selamat Datang di Platform Evaluasi Properti</h3>
            <p style="color: #64748B; font-size: 15px; line-height: 1.6; max-width: 600px; margin: 0 auto 25px auto;">
                Sistem ini siap membantu Anda memprediksi nilai jual aset rumah secara objektif di Depok. 
                Kami menggunakan pemodelan kecerdasan buatan berbasis data pasar aktual.
            </p>
            <div style="display: inline-block; background-color: #F1F5F9; padding: 12px 24px; border-radius: 30px; font-weight: 600; color: #2563EB; font-size: 14px;">
                👈 Silakan isi spesifikasi rumah di menu kiri untuk memulai
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------- TAB 2: TRANSPARANSI ALGORITMA MODEL ----------
with tab2:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0px; color:#0F172A; font-weight:700;'>📊 Transparansi Formula Matematika (OLS)</h3>", unsafe_allow_html=True)
    st.write("Di bawah ini merupakan bobot pengaruh konkrit (*Beta Coefficient*) yang dimiliki tiap variabel penentu terhadap nilai akhir harga rumah.")
    
    c_table, c_math = st.columns([1, 1])
    
    with c_table:
        st.markdown("**Bobot Pengaruh Fitur (*Feature Importance*):**")
        df_koef = pd.DataFrame({
            'Karakteristik Properti': FITUR,
            'Nilai Koefisien (Bobot)': model.coef_.round(4),
        })
        st.dataframe(df_koef, use_container_width=True, hide_index=True)
        
    with c_math:
        st.markdown("**Nilai Konstanta Dasar Sistem:**")
        st.help(f"**Intercept Base (β₀):** {model.intercept_:,.2f}")
        st.markdown("""
            * **Interpretasi Nilai:** Setiap kenaikan bobot positif menandakan fitur tersebut meningkatkan nilai harga properti secara signifikan pada model latih.
            * **Standardisasi:** Seluruh perhitungan fitur diproses terlebih dahulu menggunakan *RobustScaler* untuk meredam pencilan (*outliers*) data pasar.
        """)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Footer Premium Institutional ----------
st.markdown("<br><br><hr style='border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)
col_f1, col_f2 = st.columns([1, 1])
with col_f1:
    st.markdown("<div style='color: #94A3B8; font-size: 13px; font-weight: 500;'>© 2026 Ahmad Faisal. All Rights Reserved.</div>", unsafe_allow_html=True)
with col_f2:
    st.markdown("<div style='text-align: right; color: #94A3B8; font-size: 13px; font-weight: 600;'>PPKD Jakarta Selatan — Pendidikan Kejuruan Data Analyst</div>", unsafe_allow_html=True)