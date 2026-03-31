import numpy as np
import pandas as pd
from datetime import datetime

# Simple sensor configuration
SENSOR_COLS = [f'sensor_{i:02d}' for i in range(50)]
N_SENSORS = len(SENSOR_COLS)

def generate_row():
    """Generate one row of sensor data with optional anomalies and missing values"""
    values = np.random.normal(0, 1, N_SENSORS)
    
    # Add anomalies (2% chance)
    if np.random.rand() < 0.02:
        idx = np.random.randint(0, N_SENSORS)
        values[idx] += np.random.uniform(5, 10)
    
    # Add missing values (3% chance per sensor)
    for i in range(N_SENSORS):
        if np.random.rand() < 0.03:
            values[i] = np.nan
    
    row = {col: val for col, val in zip(SENSOR_COLS, values)}
    row['timestamp'] = datetime.utcnow()
    return pd.DataFrame([row])

def stream_data(buffer, new_row):
    """Append new row and keep last 3600 rows"""
    updated = pd.concat([buffer, new_row], ignore_index=True)
    return updated.tail(3_600).reset_index(drop=True)

def clean_data(df):
    """Fill missing values"""
    df = df.copy()
    df[SENSOR_COLS] = df[SENSOR_COLS].fillna(df[SENSOR_COLS].mean())
    return df
