
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.getcwd())

# Mock the bus before importing agent
import core.bus
core.bus.get_bus = MagicMock()

from agents_code.agent3_ml.filter import MLFilterAgent
from config.settings import ML_MODELS_DIR, LIVE_TIMEFRAME

def test_ml_agent_loading():
    print(f"LIVE_TIMEFRAME: {LIVE_TIMEFRAME}")
    print(f"ML_MODELS_DIR: {ML_MODELS_DIR}")
    
    primary_path = Path(ML_MODELS_DIR) / f"nifty_{LIVE_TIMEFRAME}.pkl"
    
    print(f"Primary path: {primary_path} (exists: {primary_path.exists()})")
    
    agent = MLFilterAgent(enable_file_watcher=False)
    
    print(f"Agent model path: {agent._model_path}")
    print(f"Ensemble is_trained: {agent.ensemble.is_trained}")
    
    if agent.ensemble.is_trained:
        print("SUCCESS: ML model loaded successfully from configured timeframe.")
    else:
        print("FAILURE: ML Model not loaded.")

if __name__ == "__main__":
    test_ml_agent_loading()
