```python
import streamlit as st
import pandas as pd
import numpy as np
from groq import Groq
import json
import os
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
import yfinance as yf

# 📄 PDF İçin Gerekli ReportLab Modülleri
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# 🔤 Font Kayıt Modülleri
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Sayfa Yapılandırması
st.set_page_config(page_title="Karar Destek ve Raporlama Platformu 3.2", layout="wide")

# HAFIZA DOSYA YOLU TANIMLAMALARI
HAFIZA_DOSYASI = "asistan_hafiza.json"

# GROQ API BAĞLANTI AYARI
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = ""

client = Groq(api_key=GROQ_API_KEY)

# --- 1. HESAPLAMA VE ANALİZ MOTORLARI (FİNANS & MAKİNE ÖĞRENMESİ) ---

def analiz_haber_duygusu(hisse_kod, client):
    try:
        ticker = yf.Ticker(hisse_kod)
        haberler = ticker.news[:3]
        if not haberler:
            return 0.5
        metin_toplami = ""
        for h in haberler:
            metin_toplami += f" Başlık: {h['title']}. Özet: {h.get('publisher', '')}\n"
            
        sistem_mesaji = (
            "Sen bir finansal duygu analiz uzmanısın. Haberleri incele. "
            "Bu haberler bahsi geçen hisse için olumlu mu, olumsuz mu karar ver. "
            "Cevap olarak SADECE 0.0 ile 1.0 arasında tek bir sayısal değer dön. Açıklama yazma."
        )
        tamamlama = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": sistem_mesaji},
                {"role": "user", "content": f"Haberler:\n{metin_toplami}"}
            ],
            temperature=0.1
        )
        return float(tamamlama.choices[0].message.content.strip())
    except:
        return 0.5

def hibrit_ai_motoru(df_giris, haber_skoru=0.5):
    df = df_giris.copy()
    if len(df) < 50: return 0.5, 0
    df['EMA_Fark'] = df['Close'].ewm(span=9).mean() - df['Close'].ewm(span=21).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    df['Momentum'] = df['Close'].diff(5)
    df['Volatilite'] = df['Close'].rolling(window=10).std()
    df['Hedef'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    df.dropna(inplace=True)
    
    X = df[['EMA_Fark', 'RSI', 'Momentum', 'Volatilite']]
    y = df['Hedef']
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    
    dogruluk_skoru = (model.predict(X) == y).mean()
    teknik_ihtimal = model.predict_proba(X.tail(1))[0][1]
    return (teknik_ihtimal * 0.7) + (haber_skoru * 0.3), dogruluk_skoru

def backtest_simulasyonu(df_giris):
    df = df_giris.copy()
    if len(df) < 50: return None
    df['EMA_Fark'] = df['Close'].ewm(span=9).mean() - df['Close'].ewm(span=21).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    df.dropna(inplace=True)
    
    df['Sinyal'] = 0
    df.loc[(df['RSI'] < 40) & (df['EMA_Fark'] > 0), 'Sinyal'] = 1
    df.loc[(df['RSI'] > 65) | (df['EMA_Fark'] < 0), 'Sinyal'] = -1
    df['Mevcut_Pozisyon'] = df['Sinyal'].shift(1).ffill().fillna(0)
    
    df['Piyasa_Getirisi'] = df['Close'].pct_change()
    df['Strateji_Getirisi'] = df['Mevcut_Pozisyon'] * df['Piyasa_Getirisi']
    
    toplam_piyasa = (1 + df['Piyasa_Getirisi'].fillna(0)).prod() - 1
    toplam_strateji = (1 + df['Strateji_Getirisi'].fillna(0)).prod() - 1
    
    kumulatif = (1 + df['Strateji_Getirisi'].fillna(0)).cumprod()
    max_drawdown = ((kumulatif - kumulatif.cummax()) / kumulatif.cummax()).min()
    
    return {"Piyasa": toplam_piyasa * 100, "Strateji": toplam_strateji * 100, "DD": max_drawdown * 100}

# --- 2. İLERİ SEVİYE TEMEL ANALİZ, RİSK YÖNETİMİ VE KLİNİK METOTLAR ---

def temel_analiz_saglik_skoru(hisse_kod):
    try:
        ticker = yf.Ticker(hisse_kod)
        info = ticker.info
        fk = info.get("trailingPE", None)
        pddd = info.get("priceToBook", None)
        borc_ozkaynak = info.get("debtToEquity", None)
        kar_marji = info.get("profitMargins", None)
        
        skor = 5
        detaylar = []
        
        if fk:
            if fk < 15: 
                skor += 1
                detaylar.append("F/K Oranı Makul (<15)")
            elif fk > 35: 
                skor -= 1
                detaylar.append("F/K Oranı Yüksek (>35)")
        if pddd:
            if pddd < 3: 
                skor += 1
                detaylar.append("PD/DD Oranı Ucuz (<3)")
            elif pddd > 7: 
                skor -= 1
                detaylar.append("PD/DD Oranı Primli (>7)")
        if borc_ozkaynak:
            if borc_ozkaynak < 100: 
                skor += 1
                detaylar.append("Borç/Özkaynak Dengeli (<100)")
            elif borc_ozkaynak > 200: 
                skor -= 1
                detaylar.append("Yüksek Borçluluk Riski (>200)")
        if kar_marji:
            if kar_marji > 0.20: 
                skor += 2
                detaylar.append("Yüksek Karlılık Marjı (>%20)")
            elif kar_marji < 0.05: 
                skor -= 1
                detaylar.append("Düşük Karlılık Marjı (<%5)")
            
        skor = max(1, min(10, skor))
        return skor, detaylar if detaylar else ["Yeterli finansal veri bulunamadı."]
    except:
        return 5, ["Temel analiz verileri çekilirken hata oluştu."]

def kelly_pozisyon_boyutu(yuzde, basari_orani):
    try:
        p = yuzde / 100
        q = 1 - p
        b = 1.2
        kelly_f = (b * p - q) / b
        if kelly_f > 0:
            return min(kelly_f * 100, 25.0)
        return 0.0
    except:
        return 0.0

def teknik_atr_stop_loss(df_giris):
    try:
        df = df_giris.copy()
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(14).mean().iloc[-1]
        son_fiyat = float(df['Close'].iloc[-1])
        return son_fiyat - (2 * atr)
    except:
        return None

def gorev_oncelik_sirala(gorevler_listesi):
    try:
        if not gorevler_listesi:
            return []
        puanli_liste = []
        for g in gorevler_listesi:
            puan = 0
            if "Yüksek" in g["oncelik"]: puan = 3
            elif "Orta" in g["oncelik"]: puan = 2
            else: puan = 1
            puanli_liste.append((puan, g))
        puanli_liste.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in puanli_liste]
    except:
        return gorevler_listesi

# --- 🏥 KLİNİK ERKEN UYARI PUANLAMA SİSTEMİ (EWS) ---
def klinik_ews_skor_hesapla(nabiz, sistolik_tansiyon, spo2, ates):
    score = 0
    if nabiz < 40 or nabiz > 130: score += 3
    elif (40 <= nabiz < 50) or (111 <= nabiz <= 130): score += 2
    elif (50 <= nabiz < 60) or (101 <= nabiz <= 110): score += 1
    
    if sistolik_tansiyon < 70 or sistolik_tansiyon > 220: score += 3
    elif (70 <= sistolik_tansiyon < 80) or (161 <= sistolik_tansiyon <= 220): score += 2
    elif (80 <= sistolik_tansiyon < 100) or (141 <= sistolik_tansiyon <= 160): score += 1
    
    if spo2 < 91: score += 3
    elif 91 <= spo2 <= 93: score += 2
    elif 94 <= spo2 <= 95: score += 1
    
    if ates < 35.0 or ates > 39.1: score += 3
    elif (35.1 <= ates <= 36.0) or (38.1 <= ates <= 39.0): score += 1
    
    return score

# --- 📈 SİMÜLE FİZYOLOJİK EKG SİNYAL ÜRETİCİ VE FİLTRE ---
def simule_fizyolojik_sinyal():
    fs = 250
    t = np.linspace(0, 2, fs * 2)
    ecg_temiz = np.zeros(len(t))
    for peak_time in [0.3, 1.1, 1.9]:
        idx = int(peak_time * fs)
        ecg_temiz[idx-2:idx+3] = np.sin(np.linspace(0, np.pi, 5)) * 1.6
        ecg_temiz[idx-8:idx-5] = -0.15
        ecg_temiz[idx+5:idx+8] = -0.25
        ecg_temiz[idx+15:idx+30] = np.sin(np.linspace(0, np.pi, 15)) * 0.25
        ecg_temiz[idx-22:idx-15] = np.sin(np.linspace(0, np.pi, 7)) * 0.12
    gurultu = 0.3 * np.sin(2 * np.pi * 50 * t)
    ecg_gurultulu = ecg_temiz + gurultu
    ecg_filtreli = ecg_temiz + 0.03 * np.random.randn(len(t))
    return t, ecg_gurultulu, ecg_filtreli

# --- 💾 YEREL HAFIZA FONKSİYONLARI ---
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
    try:
        pdfmetrics.registerFont(TTFont('Arial', 'arial.ttf'))
        pdfmetrics.registerFont(TTFont('Arial-Bold', 'arialbd.ttf'))
        aktif_font = 'Arial'
        aktif_font_bold = 'Arial-Bold'
    except:
        aktif_font = 'Helvetica'
        aktif_font_bold = 'Helvetica-Bold'

    doc = SimpleDocTemplate(dosya_adi, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    hikaye = []
    styles = getSampleStyleSheet()
    
    baslik_stili = ParagraphStyle(
        'RaporBaslik',
        fontName=aktif_font_bold,
        fontSize=20,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=15,
        alignment=1
    )
    
    alt_baslik_stili = ParagraphStyle(
        'RaporAltBaslik',
        fontName=aktif_font_bold,
        fontSize=12,
        textColor=colors.HexColor('#1e3a8a'),
        spaceBefore=14,
        spaceAfter=8
    )
    
    metin_stili = ParagraphStyle(
        'RaporMetin',
        fontName=aktif_font,
        fontSize=9,
        textColor=colors.HexColor('#334155'),
        spaceAfter=8,
        leading=14
    )

    hikaye.append(Paragraph("<b>STRATEJİK SÜREÇ VE ANALİZ RAPORU</b>", baslik_stili))
    hikaye.append(Paragraph(f"<b>Modül / Senaryo:</b> {senaryo_adi}", metin_stili))
    hikaye.append(Paragraph("<b>Rapor Tarihi:</b> 2026", metin_stili))
    hikaye.append(Spacer(1, 15))
    
    hikaye.append(Paragraph("📊 Aktif Süreç & Operasyonel Veri Özeti", alt_baslik_stili))
    
    if not data_frame.empty:
        tablo_verisi = []
        header_row = [Paragraph(f"<b>{str(col)}</b>", ParagraphStyle('H', fontName=aktif_font_bold, fontSize=8, textColor=colors.whitesmoke)) for col in data_frame.columns]
        tablo_verisi.append(header_row)
        
        for _, row in data_frame.iterrows():
            data_row = [Paragraph(str(item), ParagraphStyle('D', fontName=aktif_font, fontSize=8, textColor=colors.HexColor('#334155'))) for item in row]
            tablo_verisi.append(data_row)
        
        pdf_tablo = Table(tablo_verisi, hAlign='LEFT')
        pdf_tablo.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('TOPPADDING', (0, 0), (-1, 0), 6),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ]))
        hikaye.append(pdf_tablo)
    else:
        hikaye.append(Paragraph("Veritabanında kayıtlı tablo hücresi bulunmuyor.", metin_stili))
        
    hikaye.append(Spacer(1, 15))
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
st.title("📊 Görsel ve Analitik Yapay Zeka Motoru 3.2")
st.subheader("Kalıcı Hafıza + Canlı Finansal Terminal + Akıllı Görev Planlama + 🏥 Medikal Triyaj & EKG Analizörü")

# --- SOL PANEL ---
st.sidebar.header("⚙️ Çalışma Modu")
secilen_senaryo = st.sidebar.selectbox(
    "Asistanınızın Odak Alanını Seçin:",
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

# --- MODÜL TETİKLEYİCİLERİ ---

# 1. GÜNLÜK ASİSTAN MODÜLÜ
if secilen_senaryo == "Günlük Asistan & Problem Çözücü":
    st.info("💡 Mühendislik projeleri, yazılım geliştirme ve günlük görev yönetim paneli aktif.")
    
    # GÖREV VE DEADLINE TRACKER
    st.sidebar.markdown("---")
    st.sidebar.subheader("📅 Proje & Görev Takibi")
    
    if "gorevler" not in st.session_state:
        st.session_state.gorevler = []
        
    yeni_gorev = st.sidebar.text_input("Yeni Görev / Ödev Adı:", "")
    g_oncelik = st.sidebar.selectbox("Önem Derecesi:", ["🟢 Düşük", "🟡 Orta", "🔴 Yüksek"])
    g_deadline = st.sidebar.text_input("Teslim Tarihi / Deadline (Örn: 22 Haziran):", "Bugün")
    
    if st.sidebar.button("➕ Görevi Listeye Ekle") and yeni_gorev:
        st.session_state.gorevler.append({"ad": yeni_gorev, "oncelik": g_oncelik, "tarih": g_deadline})
        st.sidebar.success("Görev başarıyla eklendi!")
        st.rerun()
        
    if st.session_state.gorevler:
        st.write("### 📋 Öncelikli Proje ve Görev Planı")
        sirali_gorevler = gorev_oncelik_sirala(st.session_state.gorevler)
        df_aktif_rapor_icin = pd.DataFrame(sirali_gorevler)
        df_aktif_rapor_icin.columns = ["Görev Adı", "Önem Derecesi", "Teslim Tarihi"]
        st.dataframe(df_aktif_rapor_icin, use_container_width=True)
        
        if st.button("🗑️ Tüm Görev Listesini Temizle"):
            st.session_state.gorevler = []
            st.rerun()
    else:
        df_aktif_rapor_icin = pd.DataFrame({"Durum": ["Planlanan görev yok"], "Hafıza": ["Aktif"]})

    # KOD HATA AYIKLAYICI (BUG FIXER)
    st.write("### 🛠️ Mühendislik & Yazılım Kod Analizörü")
    with st.expander("💻 Hatalı Kod Bloğunu İncelemeye Gönder"):
        hatali_kod = st.text_area("Hata aldığınız kod bloğunu buraya yapıştırın:", height=150)
        hata_mesaji = st.text_input("Aldığınız hata mesajı (varsa):", "")
        
        if st.button("🔍 Kodu Analiz Et ve Düzelten Çözümü Üret") and hatali_kod:
            with st.spinner("Yapılan mantık ve yazılım hataları ayıklanıyor..."):
                kod_prompt = f"Şu kodu: {hatali_kod} ve alınan şu hatayı: {hata_mesaji} inceleyip çözüm üret."
                analiz_response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "Sen kıdemli bir yazılım mühendisi ve kod hata ayıklama uzmanısın."},
                        {"role": "user", "content": kod_prompt}
                    ],
                    temperature=0.2
                )
                st.markdown("#### 🟢 Yapay Zeka Çözüm Raporu")
                st.write(analiz_response.choices[0].message.content)

    sistem_talimati = "Sen kullanıcının analitik düşünen, zeki kişisel yapay zeka asistanı, yazılım uzmanı ve akıl hocasısın."
    asistan_selamlama = "Merhaba! Günlük asistan ve mühendislik modülü aktif. Görevlerinizi sol panelden planlayabilir, hatalı kodlarınızı yukarıdaki analizöre bırakabilirsiniz. Bugün neyi çözelim?"

# 2. FİNANSAL PERFORMANS MODÜLÜ
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

# 3. MEDİKAL TRİYAJ & KLİNİK SÜREÇ MODÜLÜ
else:
    st.info("🚨 Biyomedikal veri analizi, klinik süreç takibi ve acil triyaj optimizasyon paneli aktif.")
    
    # MEDİKAL VİTAL PARAMETRE PANELİ
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏥 Canlı Hayati Bulgular")
    nabiz_input = st.sidebar.number_input("Nabız (BPM - Kalp Hızı):", min_value=30, max_value=220, value=75)
    sistolik_input = st.sidebar.number_input("Sistolik Kan Basıncı (Büyük Tansiyon):", min_value=50, max_value=250, value=120)
    spo2_input = st.sidebar.number_input("Oksijen Satürasyonu (SpO2 %):", min_value=50, max_value=100, value=97)
    ates_input = st.sidebar.number_input("Vücut Sıcaklığı (Ateş °C):", min_value=30.0, max_value=45.0, value=36.5, step=0.1)
    
    # EWS Hesabı
    ews_skor = klinik_ews_skor_hesapla(nabiz_input, sistolik_input, spo2_input, ates_input)
    
    st.write("### 🛡️ Acil Servis Triyaj ve Klinik Süreç Öngörüsü")
    
    # Triyaj Görsel Gösterimi
    c1, c2 = st.columns(2)
    with c1:
        if ews_skor >= 5:
            st.error(f"🔴 Erken Uyarı Skoru (EWS): {ews_skor} - KIRMIZI ALAN (Yüksek Risk!)")
            st.write("🩺 **ÖNERİ:** Hasta acil stabilizasyon odasına alınmalı, vital parametreler sürekli izlenmelidir.")
            triyaj_renk = "Kırmızı"
        elif 3 <= ews_skor < 5:
            st.warning(f"🟡 Erken Uyarı Skoru (EWS): {ews_skor} - SARI ALAN (Orta Risk!)")
            st.write("🩺 **ÖNERİ:** Hekim değerlendirmesi 30 dakika içinde yapılmalı, yakın takip başlatılmalıdır.")
            triyaj_renk = "Sarı"
        else:
            st.success(f"🟢 Erken Uyarı Skoru (EWS): {ews_skor} - YEŞİL ALAN (Stabil durum)")
            st.write("🩺 **ÖNERİ:** Rutin takip yeterlidir, ayakta veya stabil servis takibine uygundur.")
            triyaj_renk = "Yeşil"
            
    with c2:
        # Fizyolojik Sinyal EKG Çizdirme
        st.write("🩺 **Biyomedikal Sinyal Takip Analizörü**")
        t_vec, noisy_ecg, clean_ecg = simule_fizyolojik_sinyal()
        fig_ecg = go.Figure()
        fig_ecg.add_trace(go.Scatter(x=t_vec, y=noisy_ecg, name="Gürültülü EKG (50Hz)", line=dict(color="#ef553b", width=1.5)))
        fig_ecg.add_trace(go.Scatter(x=t_vec, y=clean_ecg, name="Filtrelenmiş EKG", line=dict(color="#00cc96", width=2)))
        fig_ecg.update_layout(template="plotly_dark", height=200, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_ecg, use_container_width=True)

    klinik_data = {
        "Parametre": ["Kalp Hızı (Nabız)", "Sistolik Tansiyon", "Oksijen Satürasyonu (SpO2)", "Ateş Value", "Erken Uyarı Skoru"],
        "Değer": [nabiz_input, sistolik_input, spo2_input, ates_input, ews_skor],
        "Durum Sınıfı": [triyaj_renk] * 5
    }
    df_klinik = pd.DataFrame(klinik_data)
    df_aktif_rapor_icin = df_klinik
    
    sistem_talimati = f"Sen bir Klinik Mühendislik ve Triyaj uzmanısın. Kullanıcının verdiği şu hayati verilere göre konuş: Nabız {nabiz_input}, Tansiyon {sistolik_input}, SpO2 {spo2_input}, Ateş {ates_input}. EWS Skoru: {ews_skor}."
    asistan_selamlama = "🚨 Klinik Süreç Modülü Aktif. Darboğazları ve hasta durumlarını konuşalım mı?"

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

# --- YENİ NESİL BORSA ANALİZ VE BACKTEST PANELİ ---
st.sidebar.divider()
st.sidebar.subheader("📈 Canlı Borsa Analiz Modülü")
hisse_kod = st.sidebar.text_input("Hisse/Varlık Kodu (Örn: THYAO.IS, BTC-USD):", "")
analiz_buton = st.sidebar.button("🤖 Yapay Zeka Analizini Başlat")

if analiz_buton and hisse_kod:
    with st.spinner(f"{hisse_kod} için veriler çekiliyor ve analiz ediliyor..."):
        data = yf.download(hisse_kod, period="6mo", interval="1d")
        
        if not data.empty:
            haber_skoru = analiz_haber_duygusu(hisse_kod, client)
            ihtimal, basari_orani = hibrit_ai_motoru(data, haber_skoru)
            yuzde = ihtimal * 100
            
            # --- YENİ NESİL GELİŞMİŞ ANALİZ ÇIKTILARI (METRİKLER) ---
            saglik_skoru, temel_detaylar = temel_analiz_saglik_skoru(hisse_kod)
            kasa_orani = kelly_pozisyon_boyutu(yuzde, basari_orani)
            stop_seviyesi = teknik_atr_stop_loss(data)
            
            st.success(f"📊 {hisse_kod} için Kurumsal Analiz Raporu Tamamlandı!")
            
            # İlk Satır: Temel Metrikler
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("AI Yükseliş Olasılığı", f"%{yuzde:.1f}")
            m2.metric("Model Başarı Oranı", f"%{basari_orani*100:.1f}")
            m3.metric("Haber Duygu Puanı", f"{haber_skoru:.2f}")
            m4.metric("Güncel Fiyat", f"{float(data['Close'].iloc[-1]):.2f}")
            
            # İkinci Satır: Risk ve Temel Analiz Metrikleri
            st.markdown("### 🛡️ Risk Yönetimi ve Şirket Sağlık Analizi")
            r1, r2, r3 = st.columns(3)
            r1.metric("Şirket Sağlık Skoru", f"{saglik_skoru}/10")
            r2.metric("Kelly Kasa Yönetimi (Max Öneri)", f"%{kasa_orani:.1f}")
            r3.metric("Matematiksel Stop-Loss (ATR)", f"{stop_seviyesi:.2f}" if stop_seviyesi else "Hesaplanamadı")
            
            # Bilanço Detayları
            with st.expander("🔍 Şirket Sağlık Analizi Rapor Detayları"):
                for detay in temel_detaylar:
                    st.write(f"- {detay}")
            
            # Gelişmiş Strateji Mesajı
            st.divider()
            if yuzde > 60 and saglik_skoru >= 6:
                st.write("🟢 **STRATEJİ: GÜÇLÜ AL** - Hem teknik/haber yapay zekası yukarı yönlü sinyal üretiyor hem de şirketin temel finansal sağlığı çok güçlü.")
            elif yuzde < 40 or saglik_skoru <= 4:
                st.write("🔴 **STRATEJİ: GÜÇLÜ SAT / UZAK DUR** - Yapay zeka düşüş bekliyor veya şirketin borçluluk/çarpan analizleri yüksek risk barındırıyor.")
            else:
                st.write("🟡 **STRATEJİ: BEKLE/TUT** - Net bir trend oluşmadı veya teknik göstergeler temel verilerle çelişiyor. İzlemede kalın.")
                
            st.divider()
            
            # Backtest Sonuçlarını Gösteriyoruz
            bt = backtest_simulasyonu(data)
            if bt:
                st.subheader("📊 Strateji Geriye Dönük Test (Backtest) Sonuçları")
                st.write("Son 6 aylık verilere göre bu yapay zeka stratejisi uygulansaydı:")
                
                b1, b2, b3 = st.columns(3)
                b1.metric("Piyasa (Al-Tut) Getirisi", f"%{bt['Piyasa']:.1f}")
                b2.metric("AI Strateji Getirisi", f"%{bt['Strateji']:.1f}")
                b3.metric("Maksimum Risk (Drawdown)", f"%{bt['DD']:.1f}")
        else:
            st.error("Veri çekilemedi. Lütfen girdiğiniz kodun doğruluğunu kontrol edin (BIST hisseleri için sonuna .IS eklemeyi unutmayın).")

# --- YAPAY ZEKA SOHBET GİRİŞ KUTUSU ---
if user_input := st.chat_input("Asistanınıza mesajınızı yazın..."):
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
            
        except Exception as e: 
            st.error(f"Bağlantı hatası: {e}")


```
