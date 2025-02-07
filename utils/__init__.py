from typing import Dict, Any
import os

def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables."""
    return {
        'scoring': {
            'weights': {
                'name_match': float(os.getenv('SCHEMA_WEIGHT_NAME_MATCH', 0.4)),
                'type_match': float(os.getenv('SCHEMA_WEIGHT_TYPE_MATCH', 0.3)),
                'relationship_score': float(os.getenv('SCHEMA_WEIGHT_RELATIONSHIP', 0.2)),
                'pattern_match': float(os.getenv('SCHEMA_WEIGHT_PATTERN', 0.1))
            },
            'thresholds': {
                'confidence': float(os.getenv('SCHEMA_CONFIDENCE_THRESHOLD', 0.7)),
                'anomaly': float(os.getenv('SCHEMA_ANOMALY_THRESHOLD', -0.5))
            }
        },
        'ml': {
            'isolation_forest': {
                'contamination': float(os.getenv('SCHEMA_ML_CONTAMINATION', 0.1)),
                'random_state': 42,
                'max_samples': 'auto'
            }
        }
    } 