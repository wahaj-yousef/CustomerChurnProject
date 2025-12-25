from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
import pandas as pd
import joblib

app = FastAPI(title="تنبؤ انسحاب المستخدمين")

# =============================
# تحميل الموديل والـ scaler
# =============================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent / "models"

model = joblib.load(MODELS_DIR / "rf_model.pkl")
scaler = joblib.load(MODELS_DIR / "scaler.pkl")
feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")


# =============================
# الصفحة الرئيسية
# =============================
@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="ar">
<head>
<meta charset="UTF-8">
<title>تنبؤ انسحاب المستخدمين</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Arabic:wght@400;700&display=swap" rel="stylesheet">

<style>
body {
    background-color: #FFFFFF;
    font-family: 'IBM Plex Arabic', Arial, sans-serif;
    padding: 40px;
    direction: rtl;
}

h1 {
    text-align: center;
    margin-bottom: 40px;
}

.features-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 35px;
    max-width: 1300px;
    margin: auto;
}

.feature-column {
    border: 1px solid #eee;
    border-radius: 10px;
    padding: 20px;
}

.feature-column h2 {
    text-align: center;
    color: #D53636;
    margin-bottom: 20px;
}

.slider-container {
    margin-bottom: 18px;
}

label {
    font-weight: bold;
    display: block;
    margin-bottom: 6px;
    text-align: center;
}

.range-wrapper {
    display: flex;
    direction: ltr;
    align-items: center;
    gap: 10px;
}

input[type=range] {
    direction: ltr;
    flex: 1;
}

.value {
    font-weight: bold;
}

button {
    display: block;
    margin: 40px auto;
    padding: 14px 40px;
    font-size: 18px;
    background-color: #D53636;
    color: white;
    border: none;
    border-radius: 8px;
    cursor: pointer;
}

#result {
    text-align: center;
    margin-top: 30px;
}

.prediction {
    font-size: 26px;
    font-weight: bold;
}

.details {
    font-size: 18px;
    margin-top: 5px;
}
</style>
</head>

<body>

<h1>تنبؤ انسحاب المستخدمين في خدمة بث موسيقا🎵</h1>

<div class="features-grid">
    <div class="feature-column" id="core_features">
        <h2>الخصائص الأساسية</h2>
    </div>

    <div class="feature-column" id="behavior_features">
        <h2>الخصائص السلوكية</h2>
    </div>

    <div class="feature-column" id="support_features">
        <h2>الخصائص المساعدة</h2>
    </div>
</div>

<button id="predict_btn">توقع الانسحاب</button>

<div id="result">
    <div class="prediction"></div>
    <div class="details"></div>
</div>

<script>
const coreFeatures = [
 ["total_sessions","إجمالي عدد الجلسات",0,300],
 ["total_listen_time","إجمالي وقت الاستماع (دقائق)",0,15000],
 ["unique_artists","عدد الفنانين المختلفين",0,800],
 ["unique_songs","عدد الأغاني المختلفة",0,2000],
 ["total_events","إجمالي الأحداث",0,3000],
 ["avg_listen_time","متوسط وقت الاستماع لكل جلسة",0,200],
 ["active_days","عدد الأيام النشطة",1,365],
 ["tenure_days","مدة الاشتراك بالأيام",1,800]
];

const behaviorFeatures = [
 ["thumbs_up_count","عدد الإعجابات",0,1000],
 ["thumbs_down_count","عدد عدم الإعجاب",0,500],
 ["add_to_playlist_count","إضافات لقائمة التشغيل",0,500],
 ["add_friend_count","عدد إضافة الأصدقاء",0,200],
 ["logout_count","عدد تسجيلات الخروج",0,500]
];

const supportFeatures = [
 ["days_since_last_activity","أيام منذ آخر نشاط",0,90],
 ["avg_events_per_session","متوسط الأحداث لكل جلسة",0,50],
 ["help_page_views","عدد زيارات صفحة المساعدة",0,200],
 ["error_rate","نسبة الأخطاء",0,1,0.01],
 ["is_paid","هل المستخدم مدفوع",0,1,1],
 ["paid_ratio","نسبة الاشتراك المدفوع",0,1,0.01],
 ["events_last_7d","الأحداث في آخر 7 أيام",0,1000],
 ["events_last_30d","الأحداث في آخر 30 يوم",0,3000],
 ["songs_last_30d","الأغاني في آخر 30 يوم",0,2000]
];

function renderFeatures(features, containerId){
    const container = document.getElementById(containerId);
    features.forEach(f => {
        const step = f[4] || 1;
        container.innerHTML += `
        <div class="slider-container">
            <label>${f[1]}: <span id="${f[0]}_val" class="value">0</span></label>
            <div class="range-wrapper">
                <span>${f[2]}</span>
                <input type="range" id="${f[0]}" min="${f[2]}" max="${f[3]}" step="${step}" value="0"
                oninput="document.getElementById('${f[0]}_val').innerText=this.value">
                <span>${f[3]}</span>
            </div>
        </div>`;
    });
}

renderFeatures(coreFeatures, "core_features");
renderFeatures(behaviorFeatures, "behavior_features");
renderFeatures(supportFeatures, "support_features");

document.getElementById("predict_btn").onclick = async () => {
    let data = {};
    [...coreFeatures, ...behaviorFeatures, ...supportFeatures].forEach(f => {
        data[f[0]] = parseFloat(document.getElementById(f[0]).value);
    });

    const response = await fetch("/predict_ajax", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(data)
    });

    const result = await response.json();
    const predElem = document.querySelector(".prediction");
    const detailsElem = document.querySelector(".details");

    if(result.churn_prob < 0.5){
        predElem.textContent = "غير منسحب";
        predElem.style.color = "green";
    } else {
        predElem.textContent = "منسحب";
        predElem.style.color = "#D53636";
    }

    detailsElem.textContent = "احتمال الانسحاب: " + (result.churn_prob*100).toFixed(2) + "%";
};
</script>

</body>
</html>
"""


# =============================
# مسار التنبؤ
# =============================
@app.post("/predict_ajax")
def predict_churn_ajax(data: dict):

    row = {f: data.get(f, 0) for f in feature_names}

    df = pd.DataFrame([row], columns=feature_names)
    X_scaled = scaler.transform(df)

    pred = int(model.predict(X_scaled)[0])
    proba = float(model.predict_proba(X_scaled)[0, 1])

    return JSONResponse({"churn_pred": pred, "churn_prob": proba})
