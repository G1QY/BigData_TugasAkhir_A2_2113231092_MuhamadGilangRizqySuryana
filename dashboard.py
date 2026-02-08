import streamlit as st
import pandas as pd
import plotly.express as px
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, trim, regexp_replace, when
import os
import sys
import json
from datetime import datetime
from io import BytesIO

st.set_page_config(
    page_title="Dashboard Kasus HIV Di Jawa Barat",
    layout="wide"
)

@st.cache_resource
def init_spark():
    spark = SparkSession.builder \
        .appName("HIV_Jabar_BigData_Analytics") \
        .master("local[*]") \
        .config("spark.ui.showConsoleProgress", "false") \
        .getOrCreate()
    return spark
try:
    spark = init_spark()
except Exception as e:
    st.error("Spark gagal dijalankan. Periksa instalasi Java dan Spark.")
    st.exception(e)
    sys.exit(1)

@st.cache_resource
def load_data():
    csv_path = os.path.join(
        os.path.dirname(__file__),
        "dinkes_od_17570_jumlah_kasus_hiv_berdasarkan_kelompok_umur_v1_data.csv"
    )
    if not os.path.exists(csv_path):
        raise FileNotFoundError("File CSV tidak ditemukan.")
    df = spark.read.csv(
        csv_path,
        header=True,
        inferSchema=True
    )
    df = df.withColumn(
    "Kelompok_Umur",
    when(trim(col("Kelompok_Umur")).rlike(">=50|≥50|â‰¥50"), "≥50")
    .when(trim(col("Kelompok_Umur")).rlike("0.?4"), "0-4")
    .when(trim(col("Kelompok_Umur")).rlike("5.?14|14-May"), "5-14")
    .when(trim(col("Kelompok_Umur")).rlike("15.?19"), "15-19")
    .when(trim(col("Kelompok_Umur")).rlike("20.?24"), "20-24")
    .when(trim(col("Kelompok_Umur")).rlike("25.?49"), "25-49")
    .otherwise(trim(col("Kelompok_Umur")))
)
    df = df \
        .withColumnRenamed("nama_kabupaten_kota", "Kabupaten_Kota") \
        .withColumnRenamed("jumlah_kasus", "Jumlah_Kasus") \
        .withColumnRenamed("kelompok_umur", "Kelompok_Umur") \
        .withColumnRenamed("jenis_kelamin", "Jenis_Kelamin") \
        .withColumnRenamed("tahun", "Tahun")
    return df
try:
    df_spark = load_data()
except Exception as e:
    st.error("Gagal memuat data.")
    st.exception(e)
    sys.exit(1)

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/99/Coat_of_arms_of_West_Java.svg/250px-Coat_of_arms_of_West_Java.svg.png", width=100)
st.sidebar.header("Filter Data")
list_tahun = [
    row["Tahun"]
    for row in (df_spark.select("Tahun")
                .distinct()
                .sort("Tahun", ascending=False)
                .collect())
]
list_kota = [
    row["Kabupaten_Kota"]
    for row in (df_spark.select("Kabupaten_Kota")
                .distinct()
                .sort("Kabupaten_Kota")
                .collect())
]

tahun_pilih = st.sidebar.selectbox("Tahun", list_tahun)
kota_pilih = st.sidebar.multiselect(
    "Kabupaten atau Kota",
    list_kota,
    default=list_kota
)
filtered_spark = df_spark.filter(
    (col("Tahun") == tahun_pilih) &
    (col("Kabupaten_Kota").isin(kota_pilih))
)
if filtered_spark.count() == 0:
    st.warning("Data tidak tersedia untuk filter tersebut.")
    st.stop()

#SECTION: Download Data
st.sidebar.divider()
st.sidebar.subheader("Download Data")

# Convert filtered Spark DataFrame to pandas for export
try:
    df_filtered_pd = filtered_spark.toPandas()
except Exception:
    df_filtered_pd = pd.DataFrame()

# Format selector (menyerupai dropdown 'Unduh' lalu pilih format)
format_choice = st.sidebar.selectbox(
    "Pilih Format Unduh",
    ("CSV", "Excel", "JSON"),
    index=0
)

# Build filename including selected year filter
date_str = datetime.now().strftime('%Y%m%d')
base_name = f"dinkes_hiv_jabar_{tahun_pilih}_filtered_{date_str}"

download_data = None
download_mime = None
download_ext = None

if format_choice == "CSV":
    csv_buffer = BytesIO()
    df_filtered_pd.to_csv(
    csv_buffer,
    index=False,
    encoding="utf-8-sig"
)
    csv_buffer.seek(0)
    download_data = csv_buffer.getvalue()
    download_mime = "text/csv"
    download_ext = "csv"
elif format_choice == "Excel":
    try:
        excel_buffer = BytesIO()
        df_filtered_pd.to_excel(excel_buffer, index=False, sheet_name="Data")
        excel_buffer.seek(0)
        download_data = excel_buffer.getvalue()
        download_mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        download_ext = "xlsx"
    except Exception:
        st.sidebar.warning("Excel download tidak tersedia untuk data yang dipilih")
        download_data = None
elif format_choice == "JSON":
    try:
        json_str = json.dumps(df_filtered_pd.to_dict(orient="records"), ensure_ascii=False, indent=2)
        download_data = json_str.encode('utf-8')
        download_mime = "application/json"
        download_ext = "json"
    except Exception:
        st.sidebar.warning("JSON download tidak tersedia")
        download_data = None

if download_data is not None and download_ext is not None:
    st.sidebar.download_button(
        label="Unduh",
        data=download_data,
        file_name=f"{base_name}.{download_ext}",
        mime=download_mime,
        key="download_filtered"
    )
else:
    st.sidebar.info("Pilih format yang tersedia untuk mengunduh data")

#Agregasi total kasus
total_kasus = filtered_spark \
    .agg(spark_sum("Jumlah_Kasus")) \
    .collect()[0][0]
if total_kasus is None:
    total_kasus = 0
#Jumlah wilayah
jumlah_wilayah = filtered_spark \
    .select("Kabupaten_Kota") \
    .distinct() \
    .count()
#Grouping berdasarkan umur
df_umur_spark = filtered_spark \
    .groupBy("Kelompok_Umur") \
    .agg(spark_sum("Jumlah_Kasus").alias("Total")) \
    .orderBy("Kelompok_Umur")
#Grouping berdasarkan gender
df_gender_spark = filtered_spark \
    .groupBy("Jenis_Kelamin") \
    .agg(spark_sum("Jumlah_Kasus").alias("Total"))
df_umur = df_umur_spark.toPandas()
df_gender = df_gender_spark.toPandas()
desired_age_order = ['0-4', '5-14', '15-19', '20-24', '25-49', '≥50']
if not df_umur.empty:
    unique_ages = df_umur['Kelompok_Umur'].astype(str).unique().tolist()
    final_order = [a for a in desired_age_order if a in unique_ages] + [a for a in unique_ages if a not in desired_age_order]
    df_umur['Kelompok_Umur'] = pd.Categorical(df_umur['Kelompok_Umur'], categories=final_order, ordered=True)
    df_umur = df_umur.sort_values('Kelompok_Umur')
    age_category_order = final_order
else:
    age_category_order = desired_age_order
    age_name = "N/A"
    age_total = 0
    age_pct = 0.0
try:
    if not df_umur.empty:
        kelompok_dominan = df_umur.loc[df_umur["Total"].idxmax()]
        age_name = kelompok_dominan["Kelompok_Umur"]
        age_total = int(kelompok_dominan["Total"])
        age_pct = (age_total / total_kasus * 100) if total_kasus else 0.0
    else:
        kelompok_dominan = None
except Exception:
    kelompok_dominan = None

st.title("Dashboard Analisis Kasus HIV Di Jawa Barat")
st.write(f"Tahun Analisis: {tahun_pilih}")
col1, col2 = st.columns(2)
#Tampilkan ringkasan wilayah yang dipilih secara otomatis
if len(kota_pilih) == len(list_kota):
    wilayah_display = "Semua Kota/Kabupaten"
else:
    n_kota = len(kota_pilih)
    if n_kota == 1:
        wilayah_display = kota_pilih[0]
    elif n_kota <= 3:
        wilayah_display = ", ".join(kota_pilih)
    else:
        wilayah_display = ", ".join(kota_pilih[:3]) + f" dan {n_kota-3} lainnya"

col1.metric("Total Kasus HIV", f"{total_kasus:,}")
col2.metric("Jumlah Wilayah", jumlah_wilayah)
col2.caption(f"Wilayah terpilih: {wilayah_display}")

st.divider()
kiri, kanan = st.columns(2)
with kiri:
    st.subheader("Distribusi Kasus Berdasarkan Kelompok Umur")
    fig_umur = px.bar(
        df_umur,
        x="Kelompok_Umur",
        y="Total",
        text_auto=True,
        color="Kelompok_Umur",
        category_orders={"Kelompok_Umur": age_category_order},
        template="plotly_white"
    )
    st.plotly_chart(fig_umur, use_container_width=True)

with kanan:
    st.subheader("Proporsi Kasus Berdasarkan Jenis Kelamin")
    fig_gender = px.pie(
        df_gender,
        names="Jenis_Kelamin",
        values="Total",
        hole=0.4,
        template="plotly_white"
    )
    st.plotly_chart(fig_gender, use_container_width=True)

st.divider()
st.subheader("Tren Kenaikan Kasus (2019-2023)")
st.markdown("Perhatikan Lonjakan Signifikan Pasca-Pandemi (2022)")

df_trend_spark = df_spark \
    .filter(col("Kabupaten_Kota").isin(kota_pilih)) \
    .groupBy("Tahun") \
    .agg(spark_sum("Jumlah_Kasus").alias("Total")) \
    .orderBy("Tahun")
try:
    df_trend = df_trend_spark.toPandas()
except Exception:
    df_trend = pd.DataFrame()
if df_trend.empty:
    st.info("Data tren tidak tersedia untuk pilihan filter saat ini.")
else:
    fig_trend = px.line(
        df_trend,
        x="Tahun",
        y="Total",
        markers=True,
        template="plotly_dark",
        line_shape="linear"
    )
    fig_trend.update_traces(line=dict(color="red", width=3), marker=dict(size=8, color="red"))
    fig_trend.update_layout(yaxis_title="Total_Kasus", xaxis_title="Tahun", showlegend=False)
    st.plotly_chart(fig_trend, use_container_width=True)
st.divider()

insight_col1, insight_col2, insight_col3 = st.columns(3)
try:
    age_name = kelompok_dominan["Kelompok_Umur"]
    age_total = int(kelompok_dominan["Total"])
    age_pct = (age_total / total_kasus * 100) if total_kasus else 0
    insight_col1.metric("Dominan - Kelompok Umur", f"{age_name}", delta=f"{age_total:,} kasus ({age_pct:.1f}% dari total)")
except Exception:
    insight_col1.info("Data umur tidak tersedia")
try:
    gender_dom = df_gender.loc[df_gender["Total"].idxmax()]
    gender_name = gender_dom["Jenis_Kelamin"]
    gender_total = int(gender_dom["Total"])
    gender_pct = (gender_total / total_kasus * 100) if total_kasus else 0
    insight_col2.metric("Dominan - Jenis Kelamin", f"{gender_name}", delta=f"{gender_total:,} kasus ({gender_pct:.1f}% dari total)")
except Exception:
    insight_col2.info("Data gender tidak tersedia")
try:
    top_row = df_trend.loc[df_trend["Total"].idxmax()]
    top_year = int(top_row["Tahun"]) if str(top_row["Tahun"]).isdigit() else top_row["Tahun"]
    top_total = int(top_row["Total"])
    sorted_trend = df_trend.sort_values("Tahun")
    years = list(sorted_trend["Tahun"])
    vals = list(sorted_trend["Total"])
    if top_year in years:
        idx = years.index(top_year)
        if idx > 0 and vals[idx-1] != 0:
            prev = int(vals[idx-1])
            yoy = (top_total - prev) / prev * 100
            yoy_text = f"{yoy:.1f}% naik dari {prev:,}"
        else:
            yoy_text = "Tidak ada data sebelumnya"
    else:
        yoy_text = "-"
    insight_col3.metric("Puncak Tahun", f"{top_year}", delta=f"{top_total:,} kasus — {yoy_text}")
except Exception:
    insight_col3.info("Data tren tidak tersedia")

st.divider()
st.header("Analisis & Interpretasi Hasil")
with st.expander("Buka Detail Analisis", expanded=True):
    tab_analisis = st.tabs(["Analisis Pola Data"])
    with tab_analisis[0]:
        age_name_safe = age_name if 'age_name' in locals() else "N/A"
        age_pct_safe = age_pct if 'age_pct' in locals() else 0.0
        gender_name_safe = gender_name if 'gender_name' in locals() else "N/A"
        gender_pct_safe = gender_pct if 'gender_pct' in locals() else 0.0
        st.markdown(f"""
        <style>
            h4 {{
                font-size: 16px;
                font-weight: bold;
                margin: 12px 0 8px 0;
                color: #1f77b4;
            }}
            p {{
                font-size: 16px;
                text-align: justify;
                margin: 8px 0;
                line-height: 1.6;
            }}
            ol {{
                font-size: 16px;
                font-weight: bold;
                margin: 8px 0;
                padding-left: 1.5em;
            }}
            ol li {{
                font-size: 16px;
                font-weight: normal;
                text-align: justify;
                margin: 6px 0;
                line-height: 1.6;
            }}
            strong {{
                font-weight: bold;
            }}
            a {{
                color: #0066cc;
                text-decoration: none;
            }}
            a:hover {{
                text-decoration: underline;
            }}
        </style>
        
        <div style="padding: 10px;">
        
        <h4>1. Interpretasi Tren Tahunan</h4>
        <p>Berdasarkan grafik tren, terlihat adanya lonjakan kasus yang signifikan pada tahun 2022. Hal ini kemungkinan besar merupakan fenomena 'gunung es' pasca-pandemi yaitu layanan diagnosis dan layanan pencegahan yang sempat terganggu kini pulih sehingga lebih banyak kasus terdeteksi. 
            Rekomendasi: perkuat deteksi dini (kampanye tes massal dan self-testing) dan pastikan proses link-to-care cepat serta terapi Antiretroviral (ART) di hari yang sama.</p>
        
        <h4>2. Analisis Demografi</h4>
        <p>Data menunjukkan bahwa mayoritas kasus terkonsentrasi pada kelompok usia produktif. Jika dilihat dari filter saat ini, kelompok umur dominan adalah <strong>{age_name_safe}</strong> (<strong>{age_pct_safe:.1f}%</strong>). 
        Rekomendasi: fokuskan intervensi pencegahan pada kelompok usia produktif (mis. 15-49 tahun) perluasan Pre-Exposure Prophylaxis (PrEP) untuk populasi berisiko, layanan ramah remaja, edukasi seksual komprehensif, dan program deteksi di tempat kerja/komunitas.</p>
        
        <h4>3. Distribusi Gender</h4>
        <p>Dominasi kasus pada <strong>{gender_name_safe}</strong> (<strong>{gender_pct_safe:.1f}%</strong>) menunjukkan perlunya pendekatan spesifik gender. Jika dominan Laki-laki: tingkatkan akses testing di tempat kerja dan targetkan intervensi ke <strong>populasi kunci</strong> yaitu kelompok dengan risiko tinggi seperti pria yang berhubungan seks dengan pria (MSM), pekerja seks, pengguna narkoba suntik, populasi transgender, narapidana, dan pasangan dari orang dengan HIV. Rekomendasi intervensi: perluasan PrEP, harm reduction (needle and syringe programmes), layanan testing komunitas/peer-led, serta layanan klinik yang ramah dan mudah diakses. Jika dominan Perempuan: perkuat skrining ibu hamil, program PPIA/PMTCT, dan akses ke pengobatan segera.</p>
        
        <h4>4. Rekomendasi Operasional dan Kebijakan</h4>
        <p><em>(berdasarkan pedoman internasional)</em></p>
        <ol>
            <li>Perkuat layanan testing dan konfirmasi (rapid tests, self-testing) serta pastikan konfirmasi dan link-to-care dengan cepat.</li>
            <li>Terapkan kebijakan 'Test and Treat' agar pasien mendapat ART segera dan meningkatkan viral suppression untuk mencegah transmisi.</li>
            <li>Perluas PMTCT (PPIA) dan tes virologi pada bayi bila diperlukan untuk mencegah penularan ibu-ke-anak.</li>
            <li>Implementasikan PrEP dan layanan harm reduction untuk populasi kunci serta program pencegahan di komunitas.</li>
            <li>Pulihkan layanan yang terganggu pasca-COVID melalui kampanye catch-up, layanan keliling, dan penguatan SDM.</li>
            <li>Perkuat sistem data dan monitoring (integrasi dengan platform nasional seperti SatuSehat), gunakan data umur/jenis kelamin untuk targeting intervensi.</li>
        </ol>
        
        <hr style="margin: 16px 0; border: 1px solid #ccc;">
        
        <p><strong>Sumber Ringkasan (Tautan Langsung):</strong></p>
        <ul style="list-style-type: none; padding-left: 0;">
            <li style="margin: 6px 0;"><a href="https://www.who.int/news-room/fact-sheets/detail/hiv-aids" target="_blank">WHO - HIV Fact Sheet</a></li>
            <li style="margin: 6px 0;"><a href="https://www.unaids.org/en/frequently-asked-questions-about-hiv-and-aids" target="_blank">UNAIDS - HIV and AIDS</a></li>
            <li style="margin: 6px 0;"><a href="https://www.kemkes.go.id/id/bangga-dan-tahu-pencegahan-hivaids" target="_blank">Kemenkes - Bangga & Tahu (Pencegahan HIV/AIDS)</a></li>
            <li style="margin: 6px 0;"><a href="https://www.kemkes.go.id/id/one-health-one-spirit-untuk-penanggulangan-hivaids-di-indonesia" target="_blank">Kemenkes - One Health One Spirit (Penanggulangan HIV/AIDS)</a></li>
        </ul>
        
        </div>
        """, unsafe_allow_html=True)
