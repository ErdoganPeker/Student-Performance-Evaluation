import os

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split

app = FastAPI()
templates = Jinja2Templates(directory="templates")

GRADE_LABELS = {0: "A", 1: "B", 2: "C", 3: "D", 4: "F"}

# ---------------------------------------------------------------------------
# Train models at module load using the real Kaggle student performance data
# ---------------------------------------------------------------------------
_dir = os.path.dirname(os.path.abspath(__file__))
_csv_path = os.path.join(_dir, "..", "Student_performance_data.csv")

_df = pd.read_csv(_csv_path)
_df = _df.drop(columns=["StudentID"])

_feature_cols = [c for c in _df.columns if c not in ("GPA", "GradeClass")]

X = _df[_feature_cols]
y_gpa = _df["GPA"]
y_class = _df["GradeClass"].astype(int)

X_train, X_test, y_gpa_train, y_gpa_test, y_class_train, y_class_test = train_test_split(
    X, y_gpa, y_class, test_size=0.2, random_state=42
)

reg_model = RandomForestRegressor(n_estimators=200, random_state=42)
reg_model.fit(X_train, y_gpa_train)
MODEL_R2 = float(r2_score(y_gpa_test, reg_model.predict(X_test)))

clf_model = RandomForestClassifier(n_estimators=200, random_state=42)
clf_model.fit(X_train, y_class_train)
MODEL_ACCURACY = float(accuracy_score(y_class_test, clf_model.predict(X_test)))

FEATURE_IMPORTANCE = sorted(
    (
        {"feature": name, "importance": float(importance)}
        for name, importance in zip(_feature_cols, reg_model.feature_importances_)
    ),
    key=lambda item: item["importance"],
    reverse=True,
)

GPA_DISTRIBUTION = _df["GPA"].to_numpy()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


class StudentData(BaseModel):
    age: int
    gender: int
    ethnicity: int
    parental_education: int
    study_time_weekly: float
    absences: int
    tutoring: int
    parental_support: int
    extracurricular: int
    sports: int
    music: int
    volunteering: int


@app.post("/predict")
async def predict(data: StudentData):
    row = pd.DataFrame(
        [
            {
                "Age": data.age,
                "Gender": data.gender,
                "Ethnicity": data.ethnicity,
                "ParentalEducation": data.parental_education,
                "StudyTimeWeekly": data.study_time_weekly,
                "Absences": data.absences,
                "Tutoring": data.tutoring,
                "ParentalSupport": data.parental_support,
                "Extracurricular": data.extracurricular,
                "Sports": data.sports,
                "Music": data.music,
                "Volunteering": data.volunteering,
            }
        ]
    )[_feature_cols]

    predicted_gpa = float(reg_model.predict(row)[0])
    predicted_gpa = min(max(predicted_gpa, 0.0), 4.0)

    predicted_grade_class = int(clf_model.predict(row)[0])
    predicted_grade_label = GRADE_LABELS.get(predicted_grade_class, "F")

    percentile = float((GPA_DISTRIBUTION < predicted_gpa).mean() * 100)

    return {
        "predicted_gpa": round(predicted_gpa, 2),
        "predicted_grade_class": predicted_grade_class,
        "predicted_grade_label": predicted_grade_label,
        "percentile": round(percentile, 1),
        "feature_importance": FEATURE_IMPORTANCE,
        "model_r2": round(MODEL_R2, 2),
        "model_accuracy": round(MODEL_ACCURACY, 2),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5007)
