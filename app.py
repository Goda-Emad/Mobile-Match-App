import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor

# ------------------------------
# 1️⃣ Page Config
# ------------------------------
st.set_page_config(
    page_title="Mobile Match AI",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# 2️⃣ Custom CSS Professional
# ------------------------------
st.markdown("""
<style>
/* Background Gradient Elegant */
.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: #F5F5F5;
    font-family: 'Segoe UI', sans-serif;
}

/* Sidebar Style */
.stSidebar {
    background-color: #1B263B;
    color: #FFFFFF;
    padding: 20px;
    border-radius: 12px;
}

/* Sidebar Headers */
.stSidebar h2, .stSidebar h3 {
    color: #FFD700;
    font-weight: bold;
}

/* Links in sidebar */
.stSidebar a {
    color: #00FFFF;
    font-weight: bold;
    text-decoration: none;
}

/* Main Headers */
h1, h2, h3 {
    color: #FFD700;
}

/* Table Style */
.stDataFrame {
    color: #FFFFFF;
}

/* Separator lines */
hr {
    border: 1px solid #FFD700;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# 3️⃣ Sidebar Info
# ------------------------------
st.sidebar.header("👨‍💻 Eng.Goda Emad")
st.sidebar.markdown("[GitHub](https://github.com/Goda-Emad)")
st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/goda-emad/)")

st.sidebar.header("⚙️ اختياراتك")
budget_min, budget_max = st.sidebar.slider(
    "حدد ميزانيتك بالـ USD",
    0, 3000, (500, 1500)
)

usage_options = ["كاميرا ممتازة", "بطارية طويلة", "أداء قوي (جيمنج)", "استخدام يومي", "دراسة / شغل"]
usage = st.sidebar.selectbox("الاستخدام الرئيسي:", usage_options)

brands = st.sidebar.multiselect(
    "اختر الماركة المفضلة:", 
    ['Apple','Samsung','Xiaomi','Honor','Oppo','Vivo','Realme','POCO']
)

st.sidebar.subheader("ميزات إضافية")
pref_large_screen = st.sidebar.checkbox("شاشة كبيرة (>6.5 بوصة)")
pref_high_ram = st.sidebar.checkbox("رام ≥ 8 جيجا")
pref_camera = st.sidebar.checkbox("كاميرا خلفية ≥ 50 ميجا")

# ------------------------------
# 4️⃣ Load Data
# ------------------------------
df = pd.read_csv("data/Mobiles_Dataset_2025_WithPlaceholders.csv")

# ------------------------------
# 5️⃣ Apply Filters
# ------------------------------
temp = df[(df["Launched Price (USA)"] >= budget_min) & (df["Launched Price (USA)"] <= budget_max)]
if brands:
    temp = temp[temp["Company Name"].isin(brands)]
if pref_large_screen:
    temp = temp[temp["Screen Size"] >= 6.5]
if pref_high_ram:
    temp = temp[temp["RAM"] >= 8]
if pref_camera:
    temp = temp[temp["Back Camera"] >= 50]

# ------------------------------
# 6️⃣ Neural Network for Match Score
# ------------------------------
features = ["RAM", "Battery_Score", "Camera_Score", "Performance_Score", "Screen Size", "Is_New_Model", "Value_Score"]
if not temp.empty:
    X = temp[features]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y = temp["Value_Score"].values

    model = MLPRegressor(hidden_layer_sizes=(128,64,32),
                         activation='relu',
                         max_iter=500,
                         random_state=42)
    model.fit(X_scaled, y)
    temp["Match_Score"] = model.predict(X_scaled)
else:
    temp["Match_Score"] = []

# ------------------------------
# 7️⃣ Display Top 10 Recommendations without images
# ------------------------------
st.title("📱 أفضل الموبايلات لك حسب اختيارك")

top10 = temp.sort_values(by="Match_Score", ascending=False).head(10)

if top10.empty:
    st.warning("😔 لا توجد موبايلات مطابقة لاختياراتك.")
else:
    for _, row in top10.iterrows():
        st.markdown(f"### {row['Model Name']} ({row['Company Name']})")
        st.write(f"💰 السعر: ${row['Launched Price (USA)']}")
        st.write(f"🔋 Battery Score: {row['Battery_Score']}, 📸 Camera Score: {row['Camera_Score']}, 🎮 Performance Score: {row['Performance_Score']}")
        st.write(f"📏 الشاشة: {row['Screen Size']} بوصة, RAM: {row['RAM']}GB, Match Score: {row['Match_Score']:.3f}")
        st.markdown("---")

# ------------------------------
# 8️⃣ Footer
# ------------------------------
st.markdown(
    "<center>Made with ❤️ by <b>Eng.Goda Emad</b> – <a href='https://github.com/Goda-Emad'>GitHub</a> | <a href='https://www.linkedin.com/in/goda-emad/'>LinkedIn</a></center>",
    unsafe_allow_html=True
)

