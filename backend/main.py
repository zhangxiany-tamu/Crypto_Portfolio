from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn

from core.portfolio_optimizer import PortfolioOptimizer
from core.backtester import Backtester
from core.data_manager import DataManager

app = FastAPI(title="Crypto Portfolio Optimizer", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
data_manager = DataManager()
portfolio_optimizer = PortfolioOptimizer(data_manager)
backtester = Backtester(data_manager)

class PortfolioRequest(BaseModel):
    symbols: List[str]
    weights: Optional[List[float]] = None
    start_date: str
    end_date: str
    initial_capital: float = 100000
    risk_tolerance: str = "moderate"  # conservative, moderate, aggressive

class OptimizationRequest(BaseModel):
    symbols: List[str]
    start_date: str
    end_date: str
    risk_tolerance: str = "moderate"
    optimization_method: str = "mean_variance"  # mean_variance, risk_parity, black_litterman

@app.get("/")
async def root():
    return {"message": "Crypto Portfolio Optimizer API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/backtest")
async def backtest_portfolio(request: PortfolioRequest):
    try:
        if not request.weights:
            # Equal weights if not specified
            request.weights = [1.0 / len(request.symbols)] * len(request.symbols)
        
        results = await backtester.run_backtest(
            symbols=request.symbols,
            weights=request.weights,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/optimize")
async def optimize_portfolio(request: OptimizationRequest):
    try:
        results = await portfolio_optimizer.optimize(
            symbols=request.symbols,
            start_date=request.start_date,
            end_date=request.end_date,
            risk_tolerance=request.risk_tolerance,
            method=request.optimization_method
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/available_symbols")
async def get_available_symbols():
    try:
        symbols = await data_manager.get_available_symbols()
        return {"symbols": symbols}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)