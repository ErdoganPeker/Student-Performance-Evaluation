from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import uvicorn
from sklearn.ensemble import GradientBoostingRegressor

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Train model at module load
np.random.seed(42)
n = 1000
sleep = np.random.uniform(5, 10, n)
study = np.random.uniform(1, 8, n)
absences = np.random.uniform(0, 20, n)
stress = np.random.randint(0, 3, n)
internet = np.random.randint(0, 3, n)
perf = (sleep * 3 + study * 6 - absences * 1.5 - stress * 4 + internet * 3 + np.random.normal(0, 5, n))
perf = np.clip(perf, 0, 100)
X = np.column_stack([sleep, study, absences, stress, internet])
model = GradientBoostingRegressor(n_estimators=50, random_state=42)
model.fit(X, perf)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


class StudentData(BaseModel):
    sleep_hours: float
    study_hours: float
    absences: float
    stress_level: int
    internet: int


@app.post("/predict")
async def predict(data: StudentData):
    x = np.array([[data.sleep_hours, data.study_hours, data.absences, data.stress_level, data.internet]])
    p = float(model.predict(x)[0])
    grade = 'A' if p >= 90 else 'B' if p >= 75 else 'C' if p >= 60 else 'D' if p >= 50 else 'F'
    return {"performance": round(p, 1), "grade": grade}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5007)
