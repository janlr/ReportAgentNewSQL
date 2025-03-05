from typing import Dict, Any, List, Optional, Callable
from agents.database_agent import DatabaseAgent
import streamlit as st
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from datetime import datetime
from utils.config import DEFAULT_SCHEMA_CONFIG
import logging

class SchemaConfigurator:
    def __init__(self, database_agent: DatabaseAgent):
        self.database_agent = database_agent
        self.logger = logging.getLogger(__name__)
        self.config = DEFAULT_SCHEMA_CONFIG
        self.config_dir = Path("./config")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
        self.load_training_data()

    async def configure_schema(self) -> Dict[str, Any]:
        """Configure schema mapping based on database structure."""
        try:
            # Get schema information
            schema_result = await self.database_agent.process({
                "action": "get_schema_info"
            })
            
            if not schema_result.get("success"):
                raise Exception(f"Failed to get schema info: {schema_result.get('error')}")
            
            schema_info = schema_result["data"]
            
            # Initialize configuration
            config = {
                "sales": {
                    "orders_table": "",
                    "order_details_table": "",
                    "products_table": "",
                    "categories_table": "",
                    "customers_table": "",
                    "territories_table": "",
                    "column_mapping": {}
                }
            }
            
            # Find relevant tables
            for full_table_name, table_info in schema_info["tables"].items():
                table_name = table_info["name"].lower()
                columns = table_info["columns"]
                
                # Map tables based on name patterns
                if "order" in table_name and "header" in table_name:
                    config["sales"]["orders_table"] = full_table_name
                    # Map common order columns
                    for col in columns:
                        col_name = col["name"].lower()
                        if "date" in col_name and "order" in col_name:
                            config["sales"]["column_mapping"]["order_date"] = col["name"]
                        elif "orderid" in col_name.replace("_", ""):
                            config["sales"]["column_mapping"]["order_id"] = col["name"]
                
                elif "order" in table_name and "detail" in table_name:
                    config["sales"]["order_details_table"] = full_table_name
                    # Map detail columns
                    for col in columns:
                        col_name = col["name"].lower()
                        if "quantity" in col_name or "qty" in col_name:
                            config["sales"]["column_mapping"]["quantity"] = col["name"]
                        elif "price" in col_name and "unit" in col_name:
                            config["sales"]["column_mapping"]["unit_price"] = col["name"]
                        elif "total" in col_name and ("line" in col_name or "amount" in col_name):
                            config["sales"]["column_mapping"]["total_amount"] = col["name"]
                
                elif "product" in table_name and "category" in table_name:
                    config["sales"]["categories_table"] = full_table_name
                
                elif "product" in table_name and "category" not in table_name:
                    config["sales"]["products_table"] = full_table_name
                    # Map product columns
                    for col in columns:
                        col_name = col["name"].lower()
                        if "name" in col_name and "product" not in col_name:
                            config["sales"]["column_mapping"]["product_name"] = col["name"]
                        elif "productid" in col_name.replace("_", ""):
                            config["sales"]["column_mapping"]["product_id"] = col["name"]
                
                elif "customer" in table_name:
                    config["sales"]["customers_table"] = full_table_name
                    # Map customer columns
                    for col in columns:
                        col_name = col["name"].lower()
                        if "customerid" in col_name.replace("_", ""):
                            config["sales"]["column_mapping"]["customer_id"] = col["name"]
                
                elif "territory" in table_name:
                    config["sales"]["territories_table"] = full_table_name
                    # Map territory columns
                    for col in columns:
                        col_name = col["name"].lower()
                        if "territoryid" in col_name.replace("_", ""):
                            config["sales"]["column_mapping"]["territory_id"] = col["name"]
                        elif "name" in col_name:
                            config["sales"]["column_mapping"]["territory_name"] = col["name"]
                        elif "region" in col_name and "code" in col_name:
                            config["sales"]["column_mapping"]["region_code"] = col["name"]
            
            # Validate the configuration
            if not await self.database_agent.validate_schema_mapping(config):
                raise Exception("Schema mapping validation failed")
            
            return config
            
        except Exception as e:
            self.logger.error(f"Error configuring schema: {str(e)}")
            raise

    def _suggest_mappings(self, schema_info: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest schema mappings using advanced heuristics."""
        mapping = {
            'sales': {
                'tables': {},
                'column_mapping': {}
            }
        }

        # Score tables and analyze relationships
        table_scores = {}
        for table_name, table_info in schema_info.items():
            score = self._calculate_table_score(table_name, table_info)
            table_scores[table_name] = score

        relationships = self._analyze_relationships(schema_info)

        # Smart column mapping
        for table_name, table_info in schema_info.items():
            # Ensure we're working with a list of columns
            columns = table_info.get('columns', [])
            if not isinstance(columns, list):
                columns = [columns]

            column_types = self._analyze_column_types(columns)
            name_scores = self._calculate_column_name_scores(columns)
            
            # Apply weighted scoring using config weights
            weights = self.config['scoring']['weights']
            for column in columns:
                # Get column name depending on format
                col_name = column['name'] if isinstance(column, dict) else str(column)
                
                scores = {
                    'name_match': name_scores.get(col_name, 0) * weights['name_match'],
                    'type_match': column_types.get(col_name, 0) * weights['type_match'],
                    'relationship_score': relationships.get(col_name, 0) * weights['relationship_score'],
                    'pattern_match': self._match_business_patterns(col_name) * weights['pattern_match']
                }
                total_score = sum(scores.values())
                
                # Use configured confidence threshold
                if total_score > self.config['scoring']['thresholds']['confidence']:
                    best_match = self._get_best_mapping_key(col_name, scores)
                    mapping['sales']['column_mapping'][best_match] = col_name

        return mapping

    def _calculate_table_score(self, table_name: str, table_info: Dict) -> float:
        """Calculate a score for how well a table matches its suspected type."""
        try:
            score = 0.0
            name_lower = table_name.lower()
            
            # Name-based scoring
            common_words = {
                'orders': ['order', 'sale', 'transaction'],
                'products': ['product', 'item', 'inventory'],
                'customers': ['customer', 'client', 'contact']
            }
            
            for category, words in common_words.items():
                if any(word in name_lower for word in words):
                    score += 0.3
            
            # Column-based scoring
            # Safely get columns, handling different schema structures
            columns = []
            if isinstance(table_info, dict):
                columns = table_info.get('columns', [])
                if isinstance(columns, dict):
                    columns = list(columns.keys())
                elif not isinstance(columns, list):
                    columns = [str(columns)]
            
            columns_lower = [str(col).lower() if isinstance(col, str) 
                            else col.get('name', '').lower() if isinstance(col, dict)
                            else '' for col in columns]
            
            expected_columns = {
                'orders': ['date', 'total', 'customer'],
                'products': ['name', 'price', 'category'],
                'customers': ['name', 'email', 'phone']
            }
            
            for category, expected in expected_columns.items():
                matches = sum(1 for exp in expected if any(exp in col for col in columns_lower))
                score += (matches / len(expected)) * 0.4
            
            # Relationship-based scoring
            foreign_keys = table_info.get('foreign_keys', [])
            if foreign_keys:
                score += 0.3
            
            return score
            
        except Exception as e:
            st.error(f"Error in _calculate_table_score: {str(e)}")
            st.write("Table Info:", table_info)  # Debug output
            return 0.0  # Return default score on error

    def _analyze_relationships(self, schema_info: Dict[str, Any]) -> Dict[str, float]:
        """Analyze relationships between tables based on foreign keys."""
        relationships = {}
        
        for table_name, table_info in schema_info.items():
            foreign_keys = table_info.get('foreign_keys', [])
            for fk in foreign_keys:
                referred_table = fk.get('referred_table')
                if referred_table:
                    relationships[f"{table_name}.{fk['constrained_columns'][0]}"] = 0.8
                    
        return relationships

    def _match_business_patterns(self, column: str) -> float:
        """Match column against common business logic patterns."""
        column_lower = column.lower()
        
        # Define business patterns
        business_patterns = {
            'date_columns': ['date', 'created', 'modified', 'timestamp'],
            'amount_columns': ['price', 'total', 'amount', 'cost'],
            'status_columns': ['status', 'state', 'condition'],
            'quantity_columns': ['quantity', 'qty', 'count', 'number']
        }
        
        for pattern_type, pattern_list in business_patterns.items():
            if any(pattern in column_lower for pattern in pattern_list):
                return 0.9
            
        return 0.0

    def _verify_mappings_ui(self, suggested_mapping: Dict[str, Any], schema_info: Dict[str, Any]) -> Dict[str, Any]:
        """Interactive UI for verifying and modifying mappings."""
        st.header("Verify Schema Mappings")

        verified_mapping = suggested_mapping.copy()
        
        # Table mappings
        st.subheader("Table Mappings")
        all_tables = list(schema_info.keys())
        
        for table_key in ['orders_table', 'products_table', 'customers_table']:
            suggested = verified_mapping['sales'].get(table_key, '')
            verified_mapping['sales'][table_key] = st.selectbox(
                f"Select {table_key.replace('_', ' ').title()}",
                options=[''] + all_tables,
                index=all_tables.index(suggested) + 1 if suggested in all_tables else 0
            )

        # Column mappings
        st.subheader("Column Mappings")
        selected_table = st.selectbox("Select Table to View Columns", all_tables)
        if selected_table:
            # Get column names from the new schema structure
            columns = [col['name'] for col in schema_info[selected_table]['columns']]
            for mapping_key in verified_mapping['sales']['column_mapping'].keys():
                suggested = verified_mapping['sales']['column_mapping'].get(mapping_key, '')
                verified_mapping['sales']['column_mapping'][mapping_key] = st.selectbox(
                    f"Select column for {mapping_key.replace('_', ' ').title()}",
                    options=[''] + columns,
                    index=columns.index(suggested) + 1 if suggested in columns else 0
                )

        return verified_mapping

    def _load_existing_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load existing configurations from files."""
        configs = {}
        for config_file in self.config_dir.glob("*.json"):
            try:
                with open(config_file, 'r') as f:
                    configs[config_file.stem] = json.load(f)
            except Exception as e:
                st.warning(f"Error loading config {config_file.name}: {str(e)}")
        return configs

    def _save_configuration(self, name: str, config: Dict[str, Any]):
        """Save configuration to file."""
        config_path = self.config_dir / f"{name}.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def load_training_data(self):
        """Load training data for ML-based suggestions."""
        self.training_data = {
            'orders': {
                'tables': ['sales_orders', 'customer_orders', 'order_header', 'transactions'],
                'columns': ['order_date', 'order_total', 'customer_id', 'status']
            },
            'products': {
                'tables': ['products', 'items', 'inventory', 'product_catalog'],
                'columns': ['product_name', 'unit_price', 'stock_level', 'category_id']
            },
            # Add more training examples
        }

    def _calculate_ml_similarity(self, name: str, category: str, type: str) -> float:
        """Calculate similarity score using ML."""
        training_examples = self.training_data[category][type]
        if not training_examples:
            return 0.0

        # Transform text to vectors
        vectors = self.vectorizer.fit_transform([name] + training_examples)
        similarities = cosine_similarity(vectors[0:1], vectors[1:])
        return float(np.max(similarities))

    def _get_industry_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Get industry-specific schema patterns."""
        return {
            'retail': {
                'tables': {
                    'orders': ['pos_transactions', 'sales_receipts'],
                    'products': ['inventory', 'merchandise'],
                    'customers': ['shoppers', 'loyalty_members']
                },
                'columns': {
                    'price': ['unit_price', 'retail_price', 'sale_price'],
                    'quantity': ['stock_level', 'units_in_stock', 'inventory_count']
                }
            },
            'healthcare': {
                'tables': {
                    'patients': ['patient_records', 'medical_records'],
                    'appointments': ['visits', 'consultations'],
                    'treatments': ['procedures', 'medications']
                },
                'columns': {
                    'date': ['visit_date', 'admission_date', 'treatment_date'],
                    'status': ['patient_status', 'treatment_status']
                }
            },
            'finance': {
                'tables': {
                    'transactions': ['bank_transactions', 'financial_transactions'],
                    'accounts': ['bank_accounts', 'customer_accounts'],
                    'products': ['financial_products', 'banking_services']
                },
                'columns': {
                    'amount': ['transaction_amount', 'balance', 'credit_limit'],
                    'type': ['transaction_type', 'account_type']
                }
            }
        }

    def _get_enhanced_business_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Get enhanced business patterns for column mapping."""
        return {
            'temporal': {
                'date': ['date', 'timestamp', 'created_at', 'modified_at', 'updated_at'],
                'period': ['year', 'month', 'quarter', 'week', 'fiscal_period'],
                'frequency': ['daily', 'weekly', 'monthly', 'yearly']
            },
            'financial': {
                'amount': ['price', 'cost', 'amount', 'total', 'balance', 'payment'],
                'currency': ['currency_code', 'exchange_rate', 'currency_symbol'],
                'accounting': ['debit', 'credit', 'ledger', 'account_number']
            },
            'location': {
                'address': ['street', 'city', 'state', 'country', 'postal_code', 'zip'],
                'coordinates': ['latitude', 'longitude', 'geo_location', 'coordinates'],
                'regions': ['territory', 'zone', 'district', 'area']
            },
            'status': {
                'state': ['status', 'state', 'condition', 'phase'],
                'flags': ['is_active', 'is_deleted', 'is_enabled', 'is_valid'],
                'progress': ['stage', 'step', 'level', 'progress']
            },
            'metrics': {
                'quantity': ['quantity', 'count', 'units', 'pieces', 'volume'],
                'percentage': ['percent', 'ratio', 'rate', 'proportion'],
                'score': ['rating', 'score', 'rank', 'grade']
            }
        }

    def _validate_mapping(self, mapping: Dict[str, Any], schema_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate mapping with enhanced rules."""
        validation_results = []

        # Data type compatibility
        for table_name, table_info in schema_info.items():
            for column in table_info['columns']:
                if column in mapping['sales']['column_mapping'].values():
                    data_type = table_info['columns'][column]['type']
                    validation_results.append(
                        self._validate_data_type_compatibility(column, data_type)
                    )

        # Relationship integrity
        validation_results.extend(
            self._validate_relationships(mapping, schema_info)
        )

        # Required fields
        validation_results.extend(
            self._validate_required_fields(mapping)
        )

        # Business rules
        validation_results.extend(
            self._validate_business_rules(mapping, schema_info)
        )

        return validation_results

    def _validate_data_type_compatibility(self, column: str, data_type: str) -> Dict[str, Any]:
        """Validate data type compatibility."""
        expected_types = {
            'date_columns': ['datetime', 'timestamp', 'date'],
            'amount_columns': ['decimal', 'numeric', 'float', 'integer'],
            'status_columns': ['varchar', 'char', 'text'],
            'quantity_columns': ['integer', 'numeric']
        }
        # Implementation...

    def _validate_relationships(self, mapping: Dict[str, Any], schema_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate relationship integrity."""
        # Implementation...

    def _validate_business_rules(self, mapping: Dict[str, Any], schema_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate business rules."""
        # Implementation...

    def _analyze_column_types(self, columns: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze column types and return type match scores."""
        type_scores = {}
        
        type_patterns = {
            'date': {
                'types': ['datetime', 'timestamp', 'date'],
                'name_patterns': ['date', 'time', 'created', 'modified', 'timestamp']
            },
            'numeric': {
                'types': ['integer', 'decimal', 'numeric', 'float', 'double'],
                'name_patterns': ['amount', 'price', 'quantity', 'count', 'number']
            },
            'string': {
                'types': ['varchar', 'char', 'text', 'string'],
                'name_patterns': ['name', 'description', 'title', 'code', 'status']
            },
            'boolean': {
                'types': ['boolean', 'bit'],
                'name_patterns': ['is_', 'has_', 'flag', 'active', 'enabled']
            }
        }
        
        for column in columns:
            # Handle both string and dict column formats
            if isinstance(column, dict):
                col_name = column['name'].lower()
                col_type = str(column.get('type', '')).lower()
            else:
                col_name = str(column).lower()
                col_type = ''  # If we don't have type information
            
            # Calculate type match score
            score = 0.0
            
            for pattern_type, patterns in type_patterns.items():
                # Check type match if we have type information
                if col_type and any(t in col_type for t in patterns['types']):
                    score += 0.6
                
                # Check name patterns
                if any(p in col_name for p in patterns['name_patterns']):
                    score += 0.4
            
            # Store score using the column name
            type_scores[col_name if isinstance(column, str) else column['name']] = min(score, 1.0)
        
        return type_scores

    def _calculate_column_name_scores(self, columns: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate similarity scores for column names."""
        name_scores = {}
        
        # Common naming patterns for different column types
        naming_patterns = {
            'identifier': ['id', 'code', 'number', 'key'],
            'name': ['name', 'title', 'label', 'description'],
            'date': ['date', 'time', 'timestamp', 'created', 'modified'],
            'amount': ['amount', 'price', 'cost', 'total', 'sum'],
            'status': ['status', 'state', 'condition', 'phase'],
            'reference': ['ref', 'reference', 'related', 'parent']
        }
        
        for column in columns:
            # Handle both string and dict column formats
            if isinstance(column, dict):
                col_name = column['name'].lower()
            else:
                col_name = str(column).lower()
            
            max_score = 0.0
            
            # Calculate ML similarity score
            ml_score = self._calculate_ml_similarity(col_name, 'columns', 'all')
            
            # Calculate pattern match score
            pattern_score = 0.0
            for pattern_type, patterns in naming_patterns.items():
                if any(pattern in col_name for pattern in patterns):
                    pattern_score = 0.8
                    break
            
            # Combine scores with weights
            max_score = max(ml_score * 0.7 + pattern_score * 0.3, max_score)
            name_scores[col_name if isinstance(column, str) else column['name']] = max_score
        
        return name_scores

    def _analyze_foreign_keys(self, foreign_keys: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze foreign key patterns and return relationship scores."""
        fk_scores = {}
        
        for fk in foreign_keys:
            constrained_cols = fk.get('constrained_columns', [])
            referred_cols = fk.get('referred_columns', [])
            
            for col in constrained_cols:
                # Score based on relationship type
                score = 0.8  # Base score for being a foreign key
                
                # Adjust score based on naming convention
                if col.lower().endswith('_id'):
                    score += 0.1
                if 'parent' in col.lower() or 'child' in col.lower():
                    score += 0.1
                
                fk_scores[col] = min(score, 1.0)
        
        return fk_scores

    def _get_best_mapping_key(self, column: str, scores: Dict[str, float]) -> str:
        """Determine the best mapping key for a column based on scores."""
        # Common mapping keys and their patterns
        mapping_keys = {
            'order_id': ['orderid', 'order_number'],
            'order_date': ['orderdate', 'date', 'created'],
            'customer_id': ['customerid', 'client_id'],
            'product_id': ['productid', 'item_id'],
            'quantity': ['quantity', 'qty', 'amount'],
            'unit_price': ['price', 'unitprice', 'cost'],
            'total_amount': ['total', 'amount', 'sum']
        }
        
        best_score = 0.0
        best_key = None
        
        for key, patterns in mapping_keys.items():
            # Calculate pattern match score
            pattern_score = 0.0
            col_lower = column.lower()
            
            if any(pattern in col_lower for pattern in patterns):
                pattern_score = 0.8
            
            # Calculate ML similarity score
            ml_score = self._calculate_ml_similarity(column, 'columns', key)
            
            # Combine scores
            total_score = (pattern_score * 0.6) + (ml_score * 0.4)
            
            if total_score > best_score:
                best_score = total_score
                best_key = key
        
        return best_key or 'unknown'

    def _validate_required_fields(self, mapping: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate required fields are present in the mapping."""
        validation_results = []
        required_fields = {
            'orders': ['order_id', 'order_date', 'customer_id'],
            'products': ['product_id', 'product_name', 'unit_price'],
            'customers': ['customer_id', 'customer_name']
        }
        
        for table_type, fields in required_fields.items():
            for field in fields:
                if field not in mapping['sales']['column_mapping']:
                    validation_results.append({
                        'level': 'error',
                        'message': f"Required field '{field}' missing for {table_type}",
                        'field': field,
                        'table_type': table_type
                    })
        
        return validation_results

    def _validate_business_rules(self, mapping: Dict[str, Any], schema_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate business rules."""
        # Implementation...

    def _get_industry_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Get industry-specific validation rules."""
        return {
            'retail': {
                'required_tables': ['orders', 'products', 'customers', 'inventory'],
                'required_relationships': [
                    ('orders', 'customers'),
                    ('orders', 'products'),
                    ('products', 'inventory')
                ],
                'data_type_rules': {
                    'price_columns': 'decimal(10,2)',
                    'quantity_columns': 'integer',
                    'date_columns': 'datetime'
                }
            },
            'healthcare': {
                'required_tables': ['patients', 'appointments', 'treatments', 'providers'],
                'required_relationships': [
                    ('appointments', 'patients'),
                    ('treatments', 'patients'),
                    ('appointments', 'providers')
                ],
                'data_type_rules': {
                    'medical_codes': 'varchar(20)',
                    'diagnosis_codes': 'varchar(10)',
                    'visit_dates': 'datetime'
                }
            },
            'finance': {
                'required_tables': ['accounts', 'transactions', 'customers', 'products'],
                'required_relationships': [
                    ('transactions', 'accounts'),
                    ('accounts', 'customers'),
                    ('accounts', 'products')
                ],
                'data_type_rules': {
                    'amount_columns': 'decimal(18,2)',
                    'account_numbers': 'varchar(50)',
                    'transaction_dates': 'datetime'
                }
            }
        }

    def _analyze_advanced_patterns(self, schema_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze advanced patterns in schema."""
        patterns = {
            'naming_conventions': self._analyze_naming_conventions(schema_info),
            'data_hierarchies': self._detect_hierarchies(schema_info),
            'temporal_patterns': self._analyze_temporal_patterns(schema_info),
            'business_domains': self._detect_business_domains(schema_info)
        }
        
        return patterns

    def _analyze_naming_conventions(self, schema_info: Dict[str, Any]) -> Dict[str, str]:
        """Detect naming conventions used in schema."""
        conventions = {
            'snake_case': r'^[a-z]+(_[a-z]+)*$',
            'camelCase': r'^[a-z]+([A-Z][a-z]*)*$',
            'PascalCase': r'^[A-Z][a-z]+([A-Z][a-z]*)*$',
            'kebab-case': r'^[a-z]+(-[a-z]+)*$'
        }
        
        detected = {}
        for table_name, info in schema_info.items():
            for convention, pattern in conventions.items():
                if re.match(pattern, table_name):
                    detected[table_name] = convention
                    break
        
        return detected

    def _detect_hierarchies(self, schema_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect hierarchical relationships in schema."""
        hierarchies = []
        
        # Self-referential relationships
        for table_name, info in schema_info.items():
            for fk in info.get('foreign_keys', []):
                if fk['referred_table'] == table_name:
                    hierarchies.append({
                        'type': 'self_referential',
                        'table': table_name,
                        'column': fk['constrained_columns'][0]
                    })
        
        # Parent-child relationships
        # Category hierarchies
        # Organizational hierarchies
        
        return hierarchies

class CustomValidationRule:
    def __init__(self, name: str, validation_func: Callable, error_message: str):
        self.name = name
        self.validate = validation_func
        self.error_message = error_message

class ValidationRuleManager:
    def __init__(self):
        self.rules = {}
        self._load_default_rules()
    
    def add_rule(self, rule: CustomValidationRule):
        """Add a custom validation rule."""
        self.rules[rule.name] = rule
    
    def validate_schema(self, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run all validation rules on schema."""
        results = []
        for rule in self.rules.values():
            if not rule.validate(schema):
                results.append({
                    'rule': rule.name,
                    'error': rule.error_message
                })
        return results

    def _load_default_rules(self):
        """Load default validation rules."""
        self.add_rule(CustomValidationRule(
            "primary_key_rule",
            lambda s: all(self._has_primary_key(t) for t in s.values()),
            "All tables must have a primary key"
        ))
        # Add more default rules... 