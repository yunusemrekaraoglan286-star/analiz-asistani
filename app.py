import streamlit as st
import pandas as pd
import numpy as np
from groq import Groq
import json
import os
import plotly.graph_objects as go

# 📄 PDF İçin Gerekli ReportLab Modülleri
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# 🔤 Font Kayıt Modülleri
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Sayfa Yapılandırması
st.set_page_config(page_title="Karar Destek ve Raporlama Platformu 3.1", layout="wide")

# HAFIZA DOSYA YOLU TANIMLAMALARI
HAFIZA_DOSYASI = "asistan_hafiza.json"

# GROQ API BAĞLANTI AYARI
# GROQ API BAĞLANTI AYARI
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)

# --- YEREL HAFIZA FONKSİYONLARI ---
def hafiza_yukle():
    if os.path.exists(HAFIZA_DOSYASI):
        try:
            with open(HAFIZA_DOSYASI, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def hafiza_kaydet(senaryo_adi, mesajlar):
    tum_hafiza = hafiza_yukle()
    tum_hafiza[senaryo_adi] = mesajlar
    try:
        with open(HAFIZA_DOSYASI, "w", encoding="utf-8") as f:
            json.dump(tum_hafiza, f, ensure_ascii=False, indent=4)
    except Exception as e:
        st.error(f"Hafıza kaydedilirken bir hata oluştu: {e}")

# --- 📄 TÜRKÇE DESTEKLİ PDF RAPOR ÜRETİCİ MOTORU ---
def pdf_rapor_uret(senaryo_adi, data_frame, mesaj_gecmisi):
    dosya_adi = "Stratejik_Analiz_Raporu.pdf"
    
    # 🔤 Windows Sisteminden Arial Fontlarını Kaydetme (Türkçe Karakter Çözümü)
    try:
        pdfmetrics.registerFont(TTFont('Arial', 'arial.ttf'))
        pdfmetrics.registerFont(TTFont('Arial-Bold', 'arialbd.ttf'))
        aktif_font = 'Arial'
        aktif_font_bold = 'Arial-Bold'
    except:
        # Eğer Windows yolları bulamazsa standart fonta geri dön (Güvenlik önlemi)
        aktif_font = 'Helvetica'
        aktif_font_bold = 'Helvetica-Bold'

    doc = SimpleDocTemplate(dosya_adi, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    hikaye = []
    styles = getSampleStyleSheet()
    
    # Yeni Tanımlanan Türkçe Uyumlu Stiller
    baslik_stili = ParagraphStyle(
        'RaporBaslik',
        fontName=aktif_font_bold,
        fontSize=22,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=15,
        alignment=1 # Center
    )
    
    alt_baslik_stili = ParagraphStyle(
        'RaporAltBaslik',
        fontName=aktif_font_bold,
        fontSize=13,
        textColor=colors.HexColor('#1e3a8a'),
        spaceBefore=14,
        spaceAfter=8
    )
    
    metin_stili = ParagraphStyle(
        'RaporMetin',
        fontName=aktif_font,
        fontSize=10,
        textColor=colors.HexColor('#334155'),
        spaceAfter=8,
        leading=14
    )

    # 1. Başlık Ekleme
    hikaye.append(Paragraph("<b>STRATEJİK SÜREÇ VE ANALİZ RAPORU</b>", baslik_stili))
    hikaye.append(Paragraph(f"<b>Modül / Senaryo:</b> {senaryo_adi}", metin_stili))
    hikaye.append(Paragraph("<b>Rapor Tarihi:</b> 21 Mayıs 2026", metin_stili))
    hikaye.append(Spacer(1, 15))
    
    # 2. Veri Tablosunu PDF'e Ekleme
    hikaye.append(Paragraph("📊 Aktif Operasyonel / Finansal Veri Özeti", alt_baslik_stili))
    
    tablo_verisi = []
    # Header'ı ekle ve hücreleri Paragraph içine al (Hücre içi metin kayması ve font uyumu için)
    header_row = [Paragraph(f"<b>{str(col)}</b>", ParagraphStyle('H', fontName=aktif_font_bold, fontSize=9, textColor=colors.whitesmoke)) for col in data_frame.columns]
    tablo_verisi.append(header_row)
    
    for _, row in data_frame.iterrows():
        data_row = [Paragraph(str(item), ParagraphStyle('D', fontName=aktif_font, fontSize=9, textColor=colors.HexColor('#334155'))) for item in row]
        tablo_verisi.append(data_row)
    
    pdf_tablo = Table(tablo_verisi, hAlign='LEFT')
    pdf_tablo.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
    ]))
    hikaye.append(pdf_tablo)
    hikaye.append(Spacer(1, 15))
    
    # 3. AI Tarafından Üretilen Yol Haritası
    hikaye.append(Paragraph("💬 Yapay Zeka Danışman Notları ve Stratejik Yol Haritası", alt_baslik_stili))
    
    asistan_mesajlari = [m for m in mesaj_gecmisi if m["role"] == "assistant"]
    
    if asistan_mesajlari:
        son_karar = asistan_mesajlari[-1]["content"]
        temiz_karar = son_karar.replace("**", "").replace("*", "-")
        
        for paragraf in temiz_karar.split("\n"):
            if paragraf.strip():
                hikaye.append(Paragraph(paragraf, metin_stili))
    else:
        hikaye.append(Paragraph("Henüz bir stratejik görüşme kaydı bulunmamaktadır.", metin_stili))
        
    doc.build(hikaye)
    return dosya_adi

# --- STREAMLIT ARAYÜZ BAŞLANGICI ---
st.title("📊 Görsel ve Analitik Yapay Zeka Motoru 3.1")
st.subheader("Kalıcı Hafıza + Canlı Akış (Streaming) + Dinamik ML Tahmini + 📄 PDF Raporlama Sistemi")

# --- SOL PANEL ---
st.sidebar.header("⚙️ Çalışma Modu")
secilen_senaryo = st.sidebar.selectbox(
    "Asistanınızın Odağını Seçin:",
    [
        "Günlük Asistan & Problem Çözücü",
        "Finansal Performans & Büyüme", 
        "Medikal Triyaj & Klinik Süreç Optimizasyonu"
    ]
)

# Senaryo Değiştiğinde Sohbet Yönetimi
if "mevcut_senaryo" not in st.session_state or st.session_state.mevcut_senaryo != secilen_senaryo:
    st.session_state.mevcut_senaryo = secilen_senaryo
    tum_hafiza = hafiza_yukle()
    st.session_state.messages = tum_hafiza.get(secilen_senaryo, [])

df_aktif_rapor_icin = pd.DataFrame()

if secilen_senaryo == "Günlük Asistan & Problem Çözücü":
    st.info("💡 Mühendislik, yazılım veya günlük her problemi konuşabilirsiniz.")
    sistem_talimati = "Sen kullanıcının analitik düşünen, zeki kişisel yapay zeka asistanı ve akıl hocasısın."
    asistan_selamlama = "Merhaba! Hafızam tamamen beynimde aktif. Bugün hangi teknik konu veya günlük durum üzerinde çalışalım?"
    df_aktif_rapor_icin = pd.DataFrame({"Mod": ["Günlük Asistan"], "Hafıza": ["Aktif"]})

elif secilen_senaryo == "Finansal Performans & Büyüme":
    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 Finansal CSV Yükle")
    uploaded_file = st.sidebar.file_uploader("Finansal Veri Setinizi Yükleyin (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        try: df = pd.read_csv(uploaded_file)
        except: st.stop()
    else:
        df = pd.DataFrame({"Ay": ["Ocak", "Şubat", "Mart", "Nisan"], "Gelir (TL)": [120000, 140000, 135000, 160000], "Gider (TL)": [95000, 105000, 110000, 115000]})

    df["Net Kar (TL)"] = df["Gelir (TL)"] - df["Gider (TL)"]
    df["Kar Marjı (%)"] = round((df["Net Kar (TL)"] / df["Gelir (TL)"]) * 100, 2)
    df_aktif_rapor_icin = df

    st.write("### 📈 Aktif Finansal Veri Tablosu")
    st.dataframe(df, use_container_width=True)

    # ML Projeksiyon Hesaplamaları
    v_boyut = len(df)
    fit_gelir = np.polyfit(np.arange(v_boyut), df["Gelir (TL)"], 1)
    fit_gider = np.polyfit(np.arange(v_boyut), df["Gider (TL)"], 1)
    t_gelir = np.polyval(fit_gelir, np.array([v_boyut, v_boyut+1]))
    t_gider = np.polyval(fit_gider, np.array([v_boyut, v_boyut+1]))
    
    m_aylar = list(df.iloc[:, 0])
    t_aylar = m_aylar + [f"{m_aylar[-1]} +1 (T)", f"{m_aylar[-1]} +2 (T)"]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_aylar[:v_boyut], y=list(df["Gelir (TL)"]), name="Gerçekleşen Gelir", line=dict(color="#00f2ff", width=4)))
    fig.add_trace(go.Scatter(x=t_aylar[:v_boyut], y=list(df["Gider (TL)"]), name="Gerçekleşen Gider", line=dict(color="#ef553b", width=4)))
    fig.add_trace(go.Scatter(x=t_aylar[v_boyut-1:], y=[df["Gelir (TL)"].iloc[-1]] + list(t_gelir), name="Öngörülen Gelir (ML)", line=dict(color="#00f2ff", width=4, dash='dash')))
    fig.add_trace(go.Scatter(x=t_aylar[v_boyut-1:], y=[df["Gider (TL)"].iloc[-1]] + list(t_gider), name="Öngörülen Gider (ML)", line=dict(color="#ef553b", width=4, dash='dash')))
    fig.update_layout(title="🔮 Gelecek Projeksiyon Grafiği", template="plotly_dark", height=380)
    st.plotly_chart(fig, use_container_width=True)

    sistem_talimati = f"Sen uzman bir Finansal Analistsin. Mevcut gelir son ayda {df['Gelir (TL)'].iloc[-1]} TL. Gelecek projeksiyonlarına uygun aksiyon önerileri ver."
    asistan_selamlama = "Finansal Modül Aktif. Stratejimizi planlayalım mı?"

else:
    klinik_data = {"Vaka Tipi (Triyaj)": ["Kırmızı", "Sarı", "Yeşil"], "Hasta Sayısı": [18, 45, 120], "Ort. Bekleme Süresi (Dakika)": [4, 28, 65]}
    df_klinik = pd.DataFrame(klinik_data)
    df_aktif_rapor_icin = df_klinik
    st.write("### 📊 Acil Servis Raporu")
    fig_klinik = go.Figure([go.Bar(x=df_klinik["Vaka Tipi (Triyaj)"], y=df_klinik["Ort. Bekleme Süresi (Dakika)"], marker_color='#00cc96')])
    fig_klinik.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig_klinik, use_container_width=True)
    sistem_talimati = "Sen bir Klinik Mühendislik uzmanısın. Sorulara buna göre yanıt ver."
    asistan_selamlama = "🚨 Klinik Süreç Modülü Aktif. Darboğazları konuşalım mı?"

# --- 📄 SİDEBAR: PDF RAPORU OLUŞTURMA BUTONU ---
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Raporlama Katmanı")
if st.sidebar.button("📊 Profesyonel PDF Raporu Üret"):
    if len(st.session_state.messages) > 0:
        with st.sidebar.spinner("PDF Raporu Derleniyor..."):
            rapor_dosyasi = pdf_rapor_uret(secilen_senaryo, df_aktif_rapor_icin, st.session_state.messages)
            
            with open(rapor_dosyasi, "rb") as f:
                st.sidebar.download_button(
                    label="📥 PDF Raporunu Bilgisayarına İndir",
                    data=f,
                    file_name=f"{secilen_senaryo.replace(' ', '_')}_Raporu.pdf",
                    mime="application/pdf"
                )
            st.sidebar.success("Rapor başarıyla hazırlandı!")
    else:
        st.sidebar.warning("Rapor oluşturabilmek için önce yapay zeka ile konuşup bir analiz yürütmelisiniz.")

# Hafızayı Sıfırlama Butonu
if st.sidebar.button("🗑️ Bu Modülün Hafızasını Sıfırla"):
    tum_hafiza = hafiza_yukle()
    if secilen_senaryo in tum_hafiza: del tum_hafiza[secilen_senaryo]
    with open(HAFIZA_DOSYASI, "w", encoding="utf-8") as f: json.dump(tum_hafiza, f, ensure_ascii=False, indent=4)
    st.session_state.messages = []
    st.sidebar.success("Modül hafızası sıfırlandı!")
    st.rerun()

# --- ORTAK SOHBET ARAYÜZÜ ---
if len(st.session_state.messages) == 0:
    st.session_state.messages = [{"role": "assistant", "content": asistan_selamlama}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.write(msg["content"])

if user_input := st.chat_input("Mesajınızı yazın..."):
    with st.chat_message("user"): st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("assistant"):
        mesaj_alani = st.empty()
        try:
            llm_messages = [{"role": "system", "content": sistem_talimati}]
            for m in st.session_state.messages[:-1]: llm_messages.append({"role": m["role"], "content": m["content"]})
            llm_messages.append({"role": "user", "content": user_input})
            
            chat_response = client.chat.completions.create(model="llama-3.1-8b-instant", messages=llm_messages, temperature=0.4, stream=True)
            
            tam_cevap = ""
            for chunk in chat_response:
                if chunk.choices[0].delta.content is not None:
                    tam_cevap += chunk.choices[0].delta.content
                    mesaj_alani.markdown(tam_cevap + "▌")
            
            mesaj_alani.markdown(tam_cevap)
            st.session_state.messages.append({"role": "assistant", "content": tam_cevap})
            hafiza_kaydet(secilen_senaryo, st.session_state.messages)
            
        except Exception as e: st.error(f"Bağlantı hatası: {e}")
