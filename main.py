# ============================================================
# Wine Class Prediction App
# Feature Selection: 3 features only
# FastAPI + Mounted Gradio Single Server
# ============================================================

import numpy as np
import gradio as gr
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. Load Dataset
# ============================================================

wine = load_wine()

X_full = wine.data
y = wine.target

feature_names = wine.feature_names
target_names = wine.target_names


# ============================================================
# 2. Feature Selection: 3 features only
# ============================================================

selected_features = [
    "alcohol",
    "malic_acid",
    "color_intensity"
]

selected_indices = [feature_names.index(feature) for feature in selected_features]

X = X_full[:, selected_indices]

print("Selected Features:")
print(selected_features)

print("\nX shape after feature selection:")
print(X.shape)


# ============================================================
# 3. Train/Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 4. Build and Train Model
# ============================================================

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

model.fit(X_train, y_train)


# ============================================================
# 5. Model Evaluation
# ============================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))


# ============================================================
# 6. FastAPI App
# ============================================================

app = FastAPI(
    title="Wine Class Prediction Single Server",
    description="Wine class classification model using only 3 selected features",
    version="1.0"
)


class WineInput(BaseModel):
    alcohol: float
    malic_acid: float
    color_intensity: float


@app.get("/")
def root():
    return {
        "message": "Wine Class Prediction API is running",
        "gradio_url": "/gradio",
        "api_docs": "/docs"
    }


@app.post("/predict")
def predict_api(data: WineInput):
    input_data = np.array([
        [
            data.alcohol,
            data.malic_acid,
            data.color_intensity
        ]
    ])

    prediction_index = model.predict(input_data)[0]
    prediction_name = target_names[prediction_index]

    probability = model.predict_proba(input_data)[0]
    confidence = float(np.max(probability))

    return {
        "selected_features": selected_features,
        "prediction_index": int(prediction_index),
        "prediction_name": prediction_name,
        "confidence": round(confidence, 4)
    }


# ============================================================
# 7. Gradio Interface
# ============================================================

def predict_gradio(alcohol, malic_acid, color_intensity):
    input_data = np.array([
        [
            alcohol,
            malic_acid,
            color_intensity
        ]
    ])

    prediction_index = model.predict(input_data)[0]
    prediction_name = target_names[prediction_index]

    probability = model.predict_proba(input_data)[0]
    confidence = float(np.max(probability))

    result = f"""
Prediction Result

Wine Class: {prediction_name}
Class Index: {prediction_index}
Confidence: {confidence:.4f}

Used Features:
1. Alcohol: {alcohol}
2. Malic Acid: {malic_acid}
3. Color Intensity: {color_intensity}
"""

    return result


gradio_app = gr.Interface(
    fn=predict_gradio,
    inputs=[
        gr.Number(label="Alcohol"),
        gr.Number(label="Malic Acid"),
        gr.Number(label="Color Intensity")
    ],
    outputs=gr.Textbox(label="Prediction Result"),
    title="Wine Class Classification Predictor",
    description="This model predicts wine class using only 3 selected features: Alcohol, Malic Acid, and Color Intensity."
)


# ============================================================
# 8. Mount Gradio to FastAPI
# ============================================================

app = gr.mount_gradio_app(
    app,
    gradio_app,
    path="/gradio"
)


# ============================================================
# 9. Run Server
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )