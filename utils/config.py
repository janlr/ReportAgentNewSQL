"""Configuration settings for schema analysis."""

DEFAULT_SCHEMA_CONFIG = {
    'scoring': {
        'weights': {
            'name_match': 0.4,
            'type_match': 0.3,
            'relationship_score': 0.2,
            'pattern_match': 0.1
        },
        'thresholds': {
            'confidence': 0.7,
            'anomaly': -0.5
        }
    },
    'ml': {
        'isolation_forest': {
            'contamination': 0.1,
            'random_state': 42,
            'max_samples': 'auto'
        }
    },
    'validation': {
        'required_score': 0.8,
        'warning_score': 0.6
    }
} 