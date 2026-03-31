import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from scipy import stats

from data import SENSOR_COLS, N_SENSORS

class RollingZScore:
    """Improved Z-Score detector with adaptive threshold"""
    def __init__(self, window=60, threshold=3.2):
        self.window = window
        self.threshold = threshold
        self.baseline_mean = None
        self.baseline_std = None
    
    def train(self, X):
        """Train on baseline normal data"""
        self.baseline_mean = X.mean(axis=0)
        self.baseline_std = X.std(axis=0)
        self.baseline_std[self.baseline_std == 0] = 1e-9
    
    def detect(self, sensor_values):
        """Detect anomaly using Z-score"""
        if self.baseline_mean is None:
            self.baseline_mean = np.zeros(len(sensor_values))
            self.baseline_std = np.ones(len(sensor_values))
        
        z_scores = np.abs((sensor_values - self.baseline_mean) / self.baseline_std)
        max_z = float(z_scores.max())
        anomaly = max_z > self.threshold
        
        return anomaly

class ImprovedIsoForest:
    """Improved Isolation Forest with proper training"""
    def __init__(self, contamination=0.02, n_estimators=200, random_state=42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def train(self, X):
        """Train on baseline data"""
        X_scaled = self.scaler.fit_transform(X)
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X_scaled)
        self.is_trained = True
    
    def detect(self, sensor_values):
        """Detect anomaly"""
        if not self.is_trained:
            return False
        
        X_scaled = self.scaler.transform([sensor_values])
        prediction = self.model.predict(X_scaled)[0]
        return prediction == -1

class ImprovedAutoencoder:
    """Improved Autoencoder using reconstruction error and statistical analysis"""
    def __init__(self, threshold_percentile=98):
        self.threshold_percentile = threshold_percentile
        self.baseline_mean = None
        self.baseline_std = None
        self.baseline_cov = None
        self.reconstruction_errors = []
        self.threshold = None
    
    def train(self, X):
        """Train on baseline data"""
        self.baseline_mean = X.mean(axis=0)
        self.baseline_std = X.std(axis=0)
        self.baseline_std[self.baseline_std == 0] = 1e-9
        self.baseline_cov = np.cov(X.T)
        
        # Calculate reconstruction errors on baseline
        self.reconstruction_errors = []
        for row in X:
            error = np.sum(((row - self.baseline_mean) / self.baseline_std) ** 2)
            self.reconstruction_errors.append(error)
        
        self.threshold = np.percentile(self.reconstruction_errors, self.threshold_percentile)
    
    def detect(self, sensor_values):
        """Detect anomaly using reconstruction error"""
        if self.baseline_mean is None:
            return False
        
        normalized = (sensor_values - self.baseline_mean) / self.baseline_std
        reconstruction_error = np.sum(normalized ** 2)
        
        return reconstruction_error > self.threshold

class EllipticEnvelopeDetector:
    """Robust covariance-based outlier detection"""
    def __init__(self, contamination=0.02, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
    
    def train(self, X):
        """Train on baseline data"""
        X_scaled = self.scaler.fit_transform(X)
        self.model = EllipticEnvelope(
            contamination=self.contamination,
            random_state=self.random_state
        )
        self.model.fit(X_scaled)
    
    def detect(self, sensor_values):
        """Detect anomaly"""
        if self.model is None:
            return False
        
        X_scaled = self.scaler.transform([sensor_values])
        prediction = self.model.predict(X_scaled)[0]
        return prediction == -1

class AnomalyEngine:
    def __init__(self, contamination=0.05):
        self.zscore = RollingZScore(threshold=2.5)
        self.isoforest = ImprovedIsoForest(contamination=contamination)
        self.autoenc = ImprovedAutoencoder(threshold_percentile=95)
        self.elliptic = EllipticEnvelopeDetector(contamination=contamination)
        self.is_trained = False
    
    def train(self, baseline_data):
        """Train all models on baseline data"""
        X = baseline_data
        self.zscore.train(X)
        self.isoforest.train(X)
        self.autoenc.train(X)
        self.elliptic.train(X)
        self.is_trained = True
    
    def detect(self, sensor_values):
        """Detect anomaly using all methods"""
        if not self.is_trained:
            return {
                "z_anom": False,
                "if_anom": False,
                "ae_anom": False,
                "ee_anom": False,
                "ensemble": False
            }
        
        z_anom = self.zscore.detect(sensor_values)
        if_anom = self.isoforest.detect(sensor_values)
        ae_anom = self.autoenc.detect(sensor_values)
        ee_anom = self.elliptic.detect(sensor_values)
        
        # Ensemble: voting with weights (requires stronger consensus for precision)
        votes = z_anom * 1.0 + if_anom * 1.2 + ae_anom * 1.1 + ee_anom * 1.1
        ensemble = votes >= 2.5  # Higher threshold for better precision
        
        return {
            "z_anom": int(z_anom),
            "if_anom": int(if_anom),
            "ae_anom": int(ae_anom),
            "ee_anom": int(ee_anom),
            "ensemble": int(ensemble)
        }
