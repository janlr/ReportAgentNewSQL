from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from utils.config import DEFAULT_SCHEMA_CONFIG

class SchemaValidator:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or DEFAULT_SCHEMA_CONFIG
        self.anomaly_detector = IsolationForest(
            **self.config['ml']['isolation_forest']
        )
        self.pattern_learner = None
        
    def train_on_valid_schemas(self, valid_schemas: List[Dict[str, Any]]):
        """Train ML models on known valid schemas."""
        # Convert schemas to features
        features = self._extract_schema_features(valid_schemas)
        self.anomaly_detector.fit(features)
        
    def validate_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate schema using ML models."""
        features = self._extract_schema_features([schema])
        anomaly_score = self.anomaly_detector.score_samples(features)[0]
        
        return {
            'is_valid': anomaly_score > self.config['scoring']['thresholds']['anomaly'],
            'confidence': (anomaly_score + 1) / 2,
            'suggestions': self._generate_suggestions(schema, anomaly_score)
        } 