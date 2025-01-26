from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pickle
import numpy as np
import pandas as pd
import os

# Load model and artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(BASE_DIR, "best_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
features = pickle.load(open(os.path.join(BASE_DIR, "features.pkl"), "rb"))

# Initialize FastAPI
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

def preprocess_input(data):
    # Convert input dictionary into a DataFrame
    df = pd.DataFrame([data])
    
    # Ensure categorical encoding matches the model's requirements
    df = pd.get_dummies(df, columns=["location", "area_type", "size"], drop_first=True)
    
    # Add missing features with a default value of 0
    for feature in features:
        if feature not in df.columns:
            df[feature] = 0
    
    # Reorder columns to match the training data
    df = df[features]
    
    # Scale the data using the saved scaler
    scaled_data = scaler.transform(df)
    return scaled_data

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    total_sqft: str = Form(...),
    bathrooms: int = Form(...),
    balcony: int = Form(...),
    location: str = Form(...),
    area_type: str = Form(...),
    size: str = Form(...),
):
    try:
        # Convert and validate total_sqft
        total_sqft = float(total_sqft)
    except ValueError:
        return templates.TemplateResponse(
            "index.html", 
            {"request": request, "error": "Invalid value for Total Square Feet"}
        )

    # Prepare data for prediction
    input_data = {
        "total_sqft": total_sqft,
        "bath": bathrooms,
        "balcony": balcony,
        "location": location,
        "area_type": area_type,
        "size": size,
    }
    
    try:
        # Preprocess the input and make a prediction
        processed_data = preprocess_input(input_data)
        prediction = model.predict(processed_data)
        price = np.expm1(prediction[0])  # If your model uses log-transformed target
        
        # Format price and send the result to the template
        return templates.TemplateResponse(
            "index.html", 
            {"request": request, "result": f"Predicted House Price (in Lakhs): ₹{price:,.2f}"}
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html", 
            {"request": request, "error": f"An error occurred: {str(e)}"}
        )
