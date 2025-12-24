# 04_api_service_ajax_v2.py
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
        <title>تنبؤ انسحاب المستخدمين</title>
        <style>
            body { 
                background-color: beige; 
                font-family: Arial, sans-serif; 
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 50px;
            }
            h1 { color: #333; text-align: center; }
            label { display: block; margin-top: 15px; }
            input[type=range] { width: 300px; }
            .value { font-weight: bold; margin-left: 10px; }
            button { margin-top: 30px; padding: 12px 25px; font-size: 18px; cursor: pointer; }
            #sliders { display: flex; flex-direction: column; align-items: center; }
            #result { margin-top: 30px; font-size: 24px; font-weight: bold; text-align: center; }
        </style>
    </head>
    <body>
        <h1>تنبؤ انسحاب المستخدمين</h1>
        
        <div id="sliders">
            <label>إجمالي عدد الجلسات: <span id="total_sessions_val" class="value">30</span></label>
            <input type="range" id="total_sessions" min="0" max="200" value="30">

            <label>إجمالي وقت الاستماع بالدقائق: <span id="total_listen_time_val" class="value">500</span></label>
            <input type="range" id="total_listen_time" min="0" max="10000" value="500">

            <label>عدد الفنانين المختلفين: <span id="unique_artists_val" class="value">10</span></label>
            <input type="range" id="unique_artists" min="0" max="500" value="10">

            <label>عدد الأغاني المختلفة: <span id="unique_songs_val" class="value">50</span></label>
            <input type="range" id="unique_songs" min="0" max="1000" value="50">

            <label>عدد الصفحات الإيجابية: <span id="PositivePage_val" class="value">20</span></label>
            <input type="range" id="PositivePage" min="0" max="500" value="20">

            <label>عدد الصفحات السلبية: <span id="NegativePage_val" class="value">5</span></label>
            <input type="range" id="NegativePage" min="0" max="500" value="5">
        </div>
        
        <button id="predict_btn">حساب وتوقع الانسحاب</button>
        
        <div id="result"></div>
        
        <script>
            // تحديث القيم جنب السلايدر
            const sliders = ['total_sessions','total_listen_time','unique_artists','unique_songs','PositivePage','NegativePage'];
            sliders.forEach(s => {
                const slider = document.getElementById(s);
                const val = document.getElementById(s+'_val');
                slider.oninput = () => { val.textContent = slider.value; }
            });

            // AJAX POST
            document.getElementById("predict_btn").onclick = async () => {
                const data = {};
                sliders.forEach(s => { data[s] = parseFloat(document.getElementById(s).value); });
                
                const response = await fetch("/predict_ajax", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify(data)
                });
                const result = await response.json();

                // تحديد اللون والإيموجي حسب مستوى الخطر
                let color = "green";
                let emoji = "✅";
                if(result.churn_prob >= 0.7){
                    color = "red";
                    emoji = "⚠️🔥";
                } else if(result.churn_prob >= 0.4){
                    color = "orange";
                    emoji = "⚠️";
                }

                document.getElementById("result").innerHTML =
                    `<span style="color:${color}">التنبؤ: <b>${result.churn_pred==1 ? 'منسحب' : 'غير منسحب'}</b> | احتمالية الانسحاب: <b>${(result.churn_prob*100).toFixed(2)}%</b> ${emoji}</span>`;
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
