%%writefile main.py
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import joblib
import pandas as pd

# ✅ Define input schema
class InputData(BaseModel):
    industry: str
    funding_egp: float
    equity_percentage: float
    duration_months: int
    target_market: str
    business_model: str
    founder_experience_years: float
    team_size: int
    traction: str
    market_size_usd: float
    funding_usd: float
    profit: float
    repeat_purchase_rate: float
    branches_count: int
    revenue: float
    customers: int
    revenue_growth: float
    profit_margin: float
    customer_growth: float
    churn_rate: float
    operating_costs: float
    debt_to_profit_ratio: float

# ✅ Load model & preprocessor
preprocessor = joblib.load("preprocessor.pkl")

class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(RegressionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.model(x)

# Dummy input to get input dimension
dummy_input = pd.DataFrame([{
    'industry': 'Food and Beverage',
    'funding_egp': 500000,
    'equity_percentage': 20.0,
    'duration_months': 18,
    'target_market': 'Youth',
    'business_model': 'B2C',
    'founder_experience_years': 3,
    'team_size': 6,
    'traction': 'High',
    'market_size_usd': 5000000,
    'funding_usd': 15000,
    'profit': 80000,
    'repeat_purchase_rate': 0.3,
    'branches_count': 2,
    'revenue': 200000,
    'customers': 1000,
    'revenue_growth': 0.15,
    'profit_margin': 0.25,
    'customer_growth': 0.2,
    'churn_rate': 0.1,
    'operating_costs': 40000,
    'debt_to_profit_ratio': 0.5
}])
input_dim = preprocessor.transform(dummy_input).shape[1]

model = RegressionModel(input_dim)
model.load_state_dict(torch.load("startup_success_model.pt", map_location=torch.device("cpu")))
model.eval()

# ✅ FastAPI app
app = FastAPI()

@app.get("/")
def root():
    return {"message": "🚀 Roomify Model is running!"}

@app.post("/predict")
def predict(input_data: InputData):
    df = pd.DataFrame([input_data.dict()])
    processed = preprocessor.transform(df)
    tensor = torch.tensor(processed, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(tensor).item()
    return {"success_probability": round(prediction, 4)}
