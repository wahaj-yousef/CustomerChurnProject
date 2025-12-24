from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
import pandas as pd
import joblib

app = FastAPI(title="تنبؤ انسحاب المستخدمين")

# -----------------------------
# المسارات الديناميكية للنموذج وscaler
# -----------------------------
current_dir = Path(__file__).parent
models_dir = current_dir.parent / "models"
model_path = models_dir / "rf_model.pkl"
scaler_path = models_dir / "scaler.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# -----------------------------
# الصفحة الرئيسية
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>تنبؤ انسحاب المستخدمين في خدمة بث موسيقا 🎵</title>
        <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Arabic:wght@400;700&display=swap" rel="stylesheet">
        <style>
            body { 
                background-color: #FFFFFF; 
                font-family: 'IBM Plex Arabic', Arial, sans-serif; 
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 50px;
            }
            h1 { color: #333; text-align: center; margin-bottom: 40px; }
            .slider-container { margin-top: 20px; display: flex; flex-direction: column; align-items: center; }
            label { margin-bottom: 5px; font-weight: bold; }
            .description { font-size: 14px; color: #555; margin-bottom: 5px; text-align: center; max-width: 400px; }
            .range-wrapper { display: flex; width: 320px; justify-content: space-between; align-items: center; }
            input[type=range] { width: 300px; }
            .value { font-weight: bold; margin-left: 10px; }
            button { 
                margin-top: 30px; 
                padding: 12px 25px; 
                font-size: 18px; 
                cursor: pointer; 
                background-color: #D53636; 
                color: white; 
                border: none; 
                border-radius: 6px;
            }
            #result { margin-top: 30px; text-align: center; }
            #result .prediction { font-size: 26px; font-weight: bold; }
            #result .details { font-size: 18px; color: black; margin-top: 5px; }
        </style>
    </head>
    <body>
        <h1>تنبؤ انسحاب المستخدمين في خدمة بث موسيقا 🎵</h1>
        
        <div id="sliders">
            <div class="slider-container">
                <label>إجمالي عدد الجلسات: <span id="total_sessions_val" class="value">0</span></label>
                <div class="description">إجمالي عدد مرات دخول المستخدم وتفاعله مع المنصة.</div>
                <div class="range-wrapper">
                    <span>0</span>
                    <input type="range" id="total_sessions" min="0" max="200" value="0">
                    <span>200</span>
                </div>
            </div>

            <div class="slider-container">
                <label>إجمالي وقت الاستماع بالدقائق: <span id="total_listen_time_val" class="value">0</span></label>
                <div class="description">مجموع الدقائق التي استمع فيها المستخدم للأغاني على المنصة.</div>
                <div class="range-wrapper">
                    <span>0</span>
                    <input type="range" id="total_listen_time" min="0" max="10000" value="0">
                    <span>10000</span>
                </div>
            </div>

            <div class="slider-container">
                <label>عدد الفنانين المختلفين: <span id="unique_artists_val" class="value">0</span></label>
                <div class="description">عدد الفنانين المختلفين الذين استمع إليهم المستخدم.</div>
                <div class="range-wrapper">
                    <span>0</span>
                    <input type="range" id="unique_artists" min="0" max="500" value="0">
                    <span>500</span>
                </div>
            </div>

            <div class="slider-container">
                <label>عدد الأغاني المختلفة: <span id="unique_songs_val" class="value">0</span></label>
                <div class="description">عدد الأغاني الفريدة التي استمع لها المستخدم.</div>
                <div class="range-wrapper">
                    <span>0</span>
                    <input type="range" id="unique_songs" min="0" max="1000" value="0">
                    <span>1000</span>
                </div>
            </div>

            <div class="slider-container">
                <label>عدد الصفحات الإيجابية: <span id="PositivePage_val" class="value">0</span></label>
                <div class="description">مثال: صفحات تشير لتفاعل إيجابي مثل NextSong و Home.</div>
                <div class="range-wrapper">
                    <span>0</span>
                    <input type="range" id="PositivePage" min="0" max="500" value="0">
                    <span>500</span>
                </div>
            </div>

            <div class="slider-container">
                <label>عدد الصفحات السلبية: <span id="NegativePage_val" class="value">0</span></label>
                <div class="description">مثال: صفحات تشير لتفاعل سلبي مثل Logout و Cancel.</div>
                <div class="range-wrapper">
                    <span>0</span>
                    <input type="range" id="NegativePage" min="0" max="500" value="0">
                    <span>500</span>
                </div>
            </div>
        </div>
        
        <button id="predict_btn">توقع الانسحاب</button>
        
        <div id="result">
            <div class="prediction"></div>
            <div class="details"></div>
        </div>
        
        <script>
            const sliders = ['total_sessions','total_listen_time','unique_artists','unique_songs','PositivePage','NegativePage'];
            sliders.forEach(s => {
                const slider = document.getElementById(s);
                const val = document.getElementById(s+'_val');
                slider.oninput = () => { val.textContent = slider.value; }
            });

            document.getElementById("predict_btn").onclick = async () => {
                const data = {};
                sliders.forEach(s => { data[s] = parseFloat(document.getElementById(s).value); });
                
                const response = await fetch("/predict_ajax", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify(data)
                });
                const result = await response.json();

                const predElem = document.querySelector("#result .prediction");
                const detailsElem = document.querySelector("#result .details");

                if(result.churn_pred == 1){
                    predElem.textContent = "منسحب";
                    predElem.style.color = "red";
                } else {
                    predElem.textContent = "غير منسحب";
                    predElem.style.color = "green";
                }

                // السطر الثاني: فقط النسبة بدون أي نص عربي
                detailsElem.textContent = (result.churn_prob*100).toFixed(2) + "%";
            };
        </script>
    </body>
    </html>
    """

# -----------------------------
# مسار التنبؤ AJAX
# -----------------------------
@app.post("/predict_ajax")
def predict_churn_ajax(data: dict):
    df = pd.DataFrame([data])
    X_scaled = scaler.transform(df)
    pred = int(model.predict(X_scaled)[0])
    proba = float(model.predict_proba(X_scaled)[0,1])
    return JSONResponse({"churn_pred": pred, "churn_prob": proba})
