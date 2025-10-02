from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os

app = FastAPI(
    title="Unemployment Rate Prediction API",
    description="Predict unemployment rate (%) using a trained XGBoost model",
    version="1.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "xgb_unemployment_model.pkl")
model = joblib.load(model_path)

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

#allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

@app.post("/predict")
async def predict(request: Request):
    payload = await request.json()
    print("Received payload:", payload)
    try:
        X = pd.DataFrame([{
            "year": payload["year"],
            "quarter_num": payload["quarter_num"],
            "time_index_q": payload["time_index_q"],
            "sex_clean_female": payload["sex_clean_female"],
            "sex_clean_male": payload["sex_clean_male"],
            'age_group_clean_15-19': payload['age_group_clean_15-19'],
            'age_group_clean_20-24': payload['age_group_clean_20-24'],
            'age_group_clean_25-29': payload['age_group_clean_25-29'],
            'age_group_clean_30-34': payload['age_group_clean_30-34'],
            'age_group_clean_35-39': payload['age_group_clean_35-39'],
            'age_group_clean_40-49': payload['age_group_clean_40-49'],
            'age_group_clean_50-59': payload['age_group_clean_50-59'],
            'age_group_clean_60': payload['age_group_clean_60']
        }])

        # Predict
        y_pred = model.predict(X)
        prediction = round(float(y_pred[0]), 2)

        return {"predicted_unemployment_rate_pct": prediction}

    except Exception as e:
        return {"message": f"Error: {str(e)}"}