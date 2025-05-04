import os
import uvicorn
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = FastAPI()

# Ensure directories exist
os.makedirs("data/input", exist_ok=True)
os.makedirs("data/test", exist_ok=True)
os.makedirs("data/output", exist_ok=True)

# Shared global state (replace with persistent storage in production)
state = {
    "df_train": None,
    "model": None,
    "scaler": None,
    "features": None,
    "predictions_file": None
}

# Feature columns
FEATURES = [
    'Education', 'Income', 'Kidhome', 'Teenhome', 'Recency', 'AmtWines',
    'AmtFruits', 'AmtMeatProducts', 'AmtFishProducts', 'AmtSweetProducts',
    'AmtGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
    'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4',
    'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Response', 'AmtOrganic',
    'AmtNonOrganic', 'AmtReadytoEat', 'AmtCookedFoods', 'AmtEatable', 'AmtNonEatable',
    'AmtCosmetic', 'Tenure_Days', 'Tenure_Months', 'Tenure_Years', 'Marital_Status_Divorced',
    'Marital_Status_Married', 'Marital_Status_Single', 'Marital_Status_Together',
    'Marital_Status_Widow', 'Age'
]

@app.post("/upload-train/")
async def upload_train(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    df = pd.read_csv(file.file)
    df.to_csv("data/input/train.csv", index=False)
    state["df_train"] = df

    return {"message": "Training data uploaded successfully.", "columns": list(df.columns)}

@app.post("/train-model/")
async def train_model(
    n_clusters: int = 3,
    init_method: str = "k-means++",
    max_iter: int = 300,
    n_init: int = 10,
    random_state: int = 42,
    tol: float = 1e-4
):
    if state["df_train"] is None:
        raise HTTPException(status_code=400, detail="Please upload training data first.")

    df = state["df_train"]

    if not all(col in df.columns for col in FEATURES):
        raise HTTPException(status_code=400, detail="Training data missing required features.")

    data = df[FEATURES]
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    model = KMeans(
        n_clusters=n_clusters,
        init=init_method,
        max_iter=max_iter,
        n_init=n_init,
        random_state=random_state,
        tol=tol
    )
    model.fit(data_scaled)

    state["model"] = model
    state["scaler"] = scaler
    state["features"] = FEATURES

    return {"message": "Model trained successfully.", "n_clusters": n_clusters}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if state["model"] is None:
        raise HTTPException(status_code=400, detail="Please train the model first.")

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    df_test = pd.read_csv(file.file)

    if not all(col in df_test.columns for col in FEATURES):
        raise HTTPException(status_code=400, detail="Test data missing required features.")

    scaler = state["scaler"]
    model = state["model"]
    X_test_scaled = scaler.transform(df_test[FEATURES])

    clusters = model.predict(X_test_scaled)

    label_map = {0: "High Spender", 1: "Moderate Spender", 2: "Low Spender"}
    df_test['Predicted Spender Type'] = [label_map.get(cluster, f"Cluster {cluster}") for cluster in clusters]

    output_file = "data/output/predicted_output.csv"
    df_test.to_csv(output_file, index=False)
    state["predictions_file"] = output_file

    # Convert all rows to dictionary format (index → row data as key-value)
    all_records = {
        int(idx): row.dropna().to_dict()
        for idx, row in df_test.iterrows()
    }

    return {
        "message": "Prediction completed.",
        "output_file": output_file,
        "total_records": len(all_records),
        "records": all_records
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8079, reload=True)
