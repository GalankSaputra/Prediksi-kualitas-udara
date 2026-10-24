import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Prediksi Kualitas Udara', page_icon=':material/air:', layout='centered')

BASE = Path(__file__).resolve().parent
HARI = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
BULAN = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli',
         'Agustus', 'September', 'Oktober', 'November', 'Desember']


def tgl_id(d):
    d = pd.Timestamp(d)
    return f'{HARI[d.weekday()]}, {d.day} {BULAN[d.month - 1]} {d.year}'


@st.cache_resource
def muat_model():
    return joblib.load(BASE / 'model_kualitas_udara.joblib')


P = muat_model()
N = P['n_lag']
AMBANG = P['ambang']
KOTA_MODEL = set(P['kota_ke_kode'])


# ======================= Persiapan data =======================
def siapkan_data(raw):
    d = raw.copy()
    d['date'] = pd.to_datetime(d['date'], errors='coerce')
    d = d.dropna(subset=['city', 'date'])
    d['city'] = d['city'].astype(str).str.strip()
    for c in ['pm2_5_mean', 'pm10_mean', 'pm2_5_max', 'us_aqi_mean']:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.drop_duplicates(['city', 'date'], keep='last')
    return d.sort_values(['city', 'date']).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def baca_csv_upload(file_bytes):
    return siapkan_data(pd.read_csv(io.BytesIO(file_bytes)))


@st.cache_data(show_spinner=False)
def baca_csv_lokal(path, versi):
    return siapkan_data(pd.read_csv(path))


def data_dari_model():
    """Cadangan bila tidak ada CSV: hanya 7 hari terakhir yang tersimpan di file model."""
    baris = []
    for k, r in P['riwayat'].items():
        akhir = pd.Timestamp(r['tanggal'])
        for i, v in enumerate(r['pm25']):
            baris.append({'city': k, 'date': akhir - pd.Timedelta(days=i), 'pm2_5_mean': v,
                          'pm10_mean': r['pm10'] if i == 0 else np.nan})
    return siapkan_data(pd.DataFrame(baris))


@st.cache_data(show_spinner=False)
def bangun_tabel_fitur(d):
    """Fitur yang sama dengan saat pelatihan, untuk SETIAP kota dan tanggal."""
    d = d[d['city'].isin(KOTA_MODEL)].copy()
    g = d.groupby('city', sort=False)
    out = pd.DataFrame({'city': d['city'], 'date': d['date']}, index=d.index)
    for i in range(N):
        out[f'pm25_lag{i}'] = g['pm2_5_mean'].shift(i)
    out['pm10_lag0'] = d['pm10_mean']
    lags = [f'pm25_lag{i}' for i in range(N)]
    out['pm25_roll3'] = out[lags[:3]].mean(axis=1)
    out['pm25_roll7'] = out[lags].mean(axis=1)
    out['pm25_tren'] = out['pm25_lag0'] - out['pm25_lag1']
    out['bulan'] = d['date'].dt.month
    out['sin_hari'] = np.sin(2 * np.pi * d['date'].dt.dayofyear / 365.25)
    out['cos_hari'] = np.cos(2 * np.pi * d['date'].dt.dayofyear / 365.25)
    out['kota_kode'] = d['city'].map(P['kota_ke_kode'])
    # jendela harus 7 hari berurutan tanpa celah dan tanpa nilai kosong
    selisih = d['date'] - g['date'].shift(N - 1)
    out['valid'] = out[P['features']].notna().all(axis=1) & (selisih == pd.Timedelta(days=N - 1))
    # nilai aktual hari berikutnya (bila ada) untuk uji mundur
    ada = (g['date'].shift(-1) - d['date']) == pd.Timedelta(days=1)
    out['pm25_besok'] = g['pm2_5_mean'].shift(-1).where(ada)
    return out


def hitung_peluang(baris):
    return P['model'].predict_proba(baris[P['features']])[:, 1]


# ======================= Sidebar: unggah CSV =======================
st.sidebar.header(':material/sync: Perbarui Data')
file_upload = st.sidebar.file_uploader(
    'Upload CSV kualitas udara terbaru', type=['csv'],
    help='Setelah diunggah, semua tab memakai data ini selama sesi berjalan. '
         'Model tidak dilatih ulang.')

data_upload = None
if file_upload is not None:
    try:
        calon = baca_csv_upload(file_upload.getvalue())
        hilang = {'city', 'date', 'pm2_5_mean', 'pm10_mean'} - set(calon.columns)
        if hilang:
            st.sidebar.error('Kolom wajib tidak ditemukan: ' + ', '.join(sorted(hilang)))
        else:
            data_upload = calon
            st.sidebar.success(f'Data diperbarui: {len(calon):,} baris, '
                               f'sampai {calon["date"].max().date()}')
            asing = sorted(set(calon['city']) - KOTA_MODEL)
            if asing:
                st.sidebar.warning('Kota di luar model (tidak diprediksi): ' + ', '.join(asing))
    except Exception as e:
        st.sidebar.error(f'Gagal membaca CSV: {e}')

path_lokal = BASE / 'idn_air_quality_daily.csv'
if data_upload is not None:
    data, sumber = data_upload, 'CSV yang diunggah'
elif path_lokal.exists():
    data, sumber = baca_csv_lokal(str(path_lokal), path_lokal.stat().st_mtime), 'file CSV lokal'
else:
    data, sumber = data_dari_model(), 'data tersimpan di model (hanya 7 hari terakhir)'

tabel_fitur = bangun_tabel_fitur(data)
tanggal_akhir_data = data['date'].max().date()

# ======================= Tampilan =======================
st.title('Prediksi Status Kualitas Udara')
st.caption('Prediksi apakah PM2.5 rata-rata **besok** melebihi pedoman WHO '
           f'({AMBANG:.0f} µg/m³ per 24 jam), memakai algoritme Random Forest. '
           f'Data yang dipakai: **{sumber}**, sampai **{tgl_id(tanggal_akhir_data)}**.')

tab1, tab2, tab3, tab4 = st.tabs([':material/query_stats: Prediksi', ':material/location_city: Semua Kota',
                                  ':material/show_chart: Data Historis', ':material/info: Tentang Model'])

# ---------------- Tab 1: prediksi satu kota ----------------
with tab1:
    daftar_kota = sorted(set(tabel_fitur.loc[tabel_fitur['valid'], 'city']))
    if not daftar_kota:
        st.error(f'Data belum cukup: setiap kota butuh minimal {N} hari berurutan '
                 'dengan nilai PM2.5 dan PM10.')
    else:
        kota = st.selectbox('Kota', daftar_kota)
        tk = tabel_fitur[tabel_fitur['city'] == kota]
        tv = tk[tk['valid']]
        tgl_min, tgl_max = tv['date'].min().date(), tv['date'].max().date()

        tanggal = st.date_input(
            'Tanggal pengukuran terakhir ("hari ini")', value=tgl_max,
            min_value=tgl_min, max_value=tgl_max, key=f'tgl_{kota}_{tgl_max}',
            help='Nilai PM2.5 dan PM10 diambil otomatis dari data untuk tanggal ini '
                 f'dan {N - 1} hari sebelumnya.')

        baris = tv[tv['date'].dt.date == tanggal]
        if baris.empty:
            st.warning('Data untuk tanggal ini tidak lengkap (ada hari yang hilang). '
                       'Pilih tanggal lain.')
        else:
            r = baris.iloc[0]
            besok = pd.Timestamp(tanggal) + pd.Timedelta(days=1)
            p = float(hitung_peluang(baris)[0])
            terkini = (tanggal == tgl_max)

            st.subheader(f':material/calendar_month: Prediksi untuk {tgl_id(besok)}')
            if terkini:
                st.caption('Prakiraan untuk hari setelah data terakhir kota ini.')
            else:
                st.caption('Tanggal ini lebih lama dari data terakhir, jadi ditampilkan sebagai '
                           'uji mundur: prediksi dibandingkan dengan nilai aktual.')

            if p >= 0.5:
                st.error(f'**Diprediksi MELEBIHI pedoman WHO** (peluang {p:.0%})', icon=':material/warning:')
            else:
                st.success(f'**Diprediksi SESUAI pedoman WHO** (peluang melebihi {p:.0%})', icon=':material/check_circle:')
            st.progress(p)

            if pd.notna(r['pm25_besok']):
                aktual = r['pm25_besok'] > AMBANG
                benar = aktual == (p >= 0.5)
                st.info(f'Nilai aktual {tgl_id(besok)}: PM2.5 **{r["pm25_besok"]:.1f} µg/m³** '
                f'({"melebihi" if aktual else "sesuai"} pedoman WHO). '
                'Prediksi **{"benar" if benar else "salah"}**.',
                icon=':material/check_circle:' if benar else ':material/cancel:')

            st.subheader('Data masukan')
            tanggal_lag = [pd.Timestamp(tanggal) - pd.Timedelta(days=i) for i in range(N)]
            label = ['Hari ini'] + [f'H-{i}' for i in range(1, N)]
            nilai = [float(r[f'pm25_lag{i}']) for i in range(N)]
            masukan = pd.DataFrame({'Hari': label,
                                    'Tanggal': [t.date() for t in tanggal_lag],
                                    'PM2.5 (µg/m³)': nilai})
            col_a, col_b = st.columns([2, 1])
            col_a.dataframe(masukan, hide_index=True,
                            column_config={'Tanggal': st.column_config.DateColumn('Tanggal'),
                                           'PM2.5 (µg/m³)': st.column_config.NumberColumn(format='%.1f')})
            col_b.metric('PM10 hari ini', f'{r["pm10_lag0"]:.1f} µg/m³')
            col_b.metric('PM2.5 hari ini', f'{nilai[0]:.1f} µg/m³',
                         delta=f'{"di atas" if nilai[0] > AMBANG else "di bawah"} ambang {AMBANG:.0f}',
                         delta_color='inverse' if nilai[0] > AMBANG else 'normal')

            grafik = pd.DataFrame({'PM2.5': nilai[::-1],
                                   f'Ambang WHO ({AMBANG:.0f})': [AMBANG] * N},
                                  index=[t.date() for t in tanggal_lag[::-1]])
            st.line_chart(grafik)

# ---------------- Tab 2: ringkasan semua kota ----------------
with tab2:
    valid_semua = tabel_fitur[tabel_fitur['valid']]
    if valid_semua.empty:
        st.error('Data belum cukup untuk membuat prediksi.')
    else:
        terakhir = valid_semua.sort_values('date').groupby('city').tail(1).copy()
        terakhir['peluang'] = hitung_peluang(terakhir) * 100
        tgl_data = terakhir['date'].max()
        st.write('Prediksi untuk **semua kota** dari data terakhir masing-masing kota.')
        st.caption(f'Data terakhir: {tgl_id(tgl_data)}. '
                   f'Prediksi berlaku untuk **{tgl_id(tgl_data + pd.Timedelta(days=1))}**.')
        tabel = pd.DataFrame({
            'Kota': terakhir['city'],
            'Tanggal data': terakhir['date'].dt.date,
            'PM2.5 terakhir': terakhir['pm25_lag0'].round(1),
            'Peluang melebihi WHO': terakhir['peluang'],
            'Prediksi besok': np.where(terakhir['peluang'] >= 50, 'Melebihi', 'Sesuai'),
        }).sort_values('Peluang melebihi WHO', ascending=False)
        st.dataframe(tabel, hide_index=True,
                     column_config={
                         'Tanggal data': st.column_config.DateColumn('Tanggal data'),
                         'Peluang melebihi WHO': st.column_config.ProgressColumn(
                             'Peluang melebihi WHO', min_value=0, max_value=100, format='%.0f%%')})
        st.metric('Kota diprediksi melebihi WHO',
                  f"{(tabel['Prediksi besok'] == 'Melebihi').sum()} dari {len(tabel)}")
        if (terakhir['date'] < tgl_data).any():
            st.warning('Sebagian kota datanya lebih lama dari tanggal terakhir; lihat kolom "Tanggal data".')

# ---------------- Tab 3: data historis ----------------
with tab3:
    st.subheader(':material/show_chart: Riwayat Kualitas Udara')
    st.write('Unggah CSV pengukuran terbaru melalui sidebar untuk memperbarui data yang ditampilkan. '
             'Aplikasi hanya membaca file yang diunggah selama sesi ini; tidak mengambil data '
             'dari sensor atau internet.')
    st.caption(f'Sumber yang sedang digunakan: **{sumber}**')

    data_csv = data
    kota_csv = sorted(data_csv['city'].astype(str).unique().tolist())
    kota_awal = 'Palu' if 'Palu' in kota_csv else kota_csv[0]
    kota_pilihan = st.selectbox('Pilih kota untuk melihat riwayat', kota_csv,
                                index=kota_csv.index(kota_awal), key='kota_riwayat_csv')

    data_kota = data_csv[data_csv['city'].astype(str) == kota_pilihan].copy()
    tanggal_min = data_kota['date'].min().date()
    tanggal_max = data_kota['date'].max().date()
    rentang = st.date_input('Rentang tanggal', value=(tanggal_min, tanggal_max),
                            min_value=tanggal_min, max_value=tanggal_max,
                            key=f'rentang_{kota_pilihan}_{tanggal_max}')

    if isinstance(rentang, (tuple, list)) and len(rentang) == 2:
        awal, akhir = rentang
        data_tampil = data_kota[(data_kota['date'].dt.date >= awal) & (data_kota['date'].dt.date <= akhir)].copy()
    elif isinstance(rentang, (tuple, list)):
        data_tampil = data_kota[data_kota['date'].dt.date == rentang[0]].copy() if rentang else data_kota.iloc[0:0]
    else:
        data_tampil = data_kota[data_kota['date'].dt.date == rentang].copy()

    if data_tampil.empty:
        st.warning('Tidak ada data untuk kota dan rentang tanggal tersebut.')
    else:
        data_tampil = data_tampil.sort_values('date')
        terbaru = data_tampil.iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric('Jumlah catatan', f'{len(data_tampil):,}')
        if pd.notna(terbaru.get('pm2_5_mean')):
            c2.metric('PM2.5 terbaru', f"{terbaru['pm2_5_mean']:.1f} µg/m³")
        if pd.notna(terbaru.get('pm10_mean')):
            c3.metric('PM10 terbaru', f"{terbaru['pm10_mean']:.1f} µg/m³")

        grafik_data = data_tampil.set_index('date')[['pm2_5_mean', 'pm10_mean']].rename(columns={
            'pm2_5_mean': 'PM2.5 rata-rata (µg/m³)', 'pm10_mean': 'PM10 rata-rata (µg/m³)'})
        st.line_chart(grafik_data)

        kolom_tampil = [k for k in ['date', 'city', 'province', 'pm2_5_mean', 'pm2_5_max',
                                    'pm10_mean', 'us_aqi_mean', 'exceeds_who_pm25']
                        if k in data_tampil.columns]
        tabel_csv = data_tampil[kolom_tampil].sort_values('date', ascending=False)
        st.dataframe(tabel_csv, hide_index=True,
                     column_config={
                         'date': st.column_config.DateColumn('Tanggal'),
                         'pm2_5_mean': st.column_config.NumberColumn('PM2.5 rata-rata', format='%.2f'),
                         'pm2_5_max': st.column_config.NumberColumn('PM2.5 maksimum', format='%.2f'),
                         'pm10_mean': st.column_config.NumberColumn('PM10 rata-rata', format='%.2f'),
                         'us_aqi_mean': st.column_config.NumberColumn('US AQI rata-rata', format='%.1f'),
                         'exceeds_who_pm25': st.column_config.CheckboxColumn('Melebihi WHO')})
        st.download_button('Unduh data yang ditampilkan (CSV)',
                           data=tabel_csv.to_csv(index=False).encode('utf-8'),
                           file_name=f'riwayat_kualitas_udara_{kota_pilihan.lower().replace(" ", "_")}.csv',
                           mime='text/csv')
        st.info('Catatan: CSV yang diunggah memperbarui data masukan prediksi di semua tab, '
                'tetapi tidak melatih ulang model Random Forest.')

# ---------------- Tab 4: tentang model ----------------
with tab4:
    m = P['metrik']
    a, b = P['periode_data']
    st.markdown(f"""
**Data pelatihan:** kualitas udara harian {len(P['kota_ke_kode'])} kota di Indonesia.

**Tugas:** klasifikasi biner status hari berikutnya (Sesuai / Melebihi pedoman WHO PM2.5 24 jam).

**Evaluasi (data uji berdasarkan waktu):** akurasi {m['akurasi']:.1%}, F1 {m['f1']:.1%}, recall {m['recall_melebihi']:.1%}.

**Pembanding:** tebakan "besok sama dengan hari ini" mencapai akurasi {m['akurasi_baseline']:.1%}.

**Data masukan saat ini:** {sumber}, sampai {tgl_id(tanggal_akhir_data)}.

**Batasan:**
- Prediksi hanya memakai riwayat PM2.5/PM10, tanpa data cuaca atau sumber emisi.
- Model dilatih sampai {b}; CSV baru hanya memperbarui masukan, bukan melatih ulang model.
- Bukan peringatan resmi. Untuk keputusan kesehatan, rujuk BMKG/KLHK.
""")