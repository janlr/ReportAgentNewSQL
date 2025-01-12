import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union

class TestDataGenerator:
    """Utility class for generating test data for various scenarios."""
    
    @staticmethod
    def generate_sales_data(
        start_date: str = '2023-01-01',
        periods: int = 100,
        categories: List[str] = ['A', 'B', 'C'],
        regions: List[str] = ['North', 'South', 'East', 'West']
    ) -> pd.DataFrame:
        """Generate sample sales data."""
        np.random.seed(42)  # For reproducibility
        
        df = pd.DataFrame({
            'date': pd.date_range(start=start_date, periods=periods),
            'sales': np.random.normal(1000, 100, periods),
            'customers': np.random.normal(50, 10, periods),
            'category': np.random.choice(categories, periods),
            'region': np.random.choice(regions, periods)
        })
        
        # Add some seasonal patterns to sales
        df['sales'] += np.sin(np.arange(periods) * 2 * np.pi / 30) * 200
        
        return df
    
    @staticmethod
    def generate_inventory_data(
        num_products: int = 50,
        categories: List[str] = ['A', 'B', 'C']
    ) -> pd.DataFrame:
        """Generate sample inventory data."""
        np.random.seed(42)
        
        df = pd.DataFrame({
            'product_id': range(1, num_products + 1),
            'category': np.random.choice(categories, num_products),
            'stock_level': np.random.randint(0, 1000, num_products),
            'reorder_point': np.random.randint(50, 200, num_products),
            'unit_cost': np.random.uniform(10, 100, num_products)
        })
        
        # Add some logical relationships
        df['total_value'] = df['stock_level'] * df['unit_cost']
        
        return df
    
    @staticmethod
    def generate_customer_data(
        num_customers: int = 1000,
        start_date: str = '2020-01-01'
    ) -> pd.DataFrame:
        """Generate sample customer data."""
        np.random.seed(42)
        
        df = pd.DataFrame({
            'customer_id': range(1, num_customers + 1),
            'join_date': pd.date_range(start=start_date, periods=num_customers),
            'lifetime_value': np.random.exponential(1000, num_customers),
            'total_orders': np.random.poisson(10, num_customers),
            'age_group': np.random.choice(['18-25', '26-35', '36-50', '50+'], num_customers)
        })
        
        # Add some correlations
        df['average_order_value'] = df['lifetime_value'] / df['total_orders']
        
        return df
    
    @staticmethod
    def generate_time_series_data(
        start_date: str = '2023-01-01',
        periods: int = 365,
        metrics: List[str] = ['value'],
        trend: float = 0.1,
        seasonality: float = 0.2,
        noise: float = 0.1
    ) -> pd.DataFrame:
        """Generate time series data with trend, seasonality, and noise."""
        np.random.seed(42)
        
        # Generate time index
        dates = pd.date_range(start=start_date, periods=periods)
        
        # Generate components
        t = np.arange(periods)
        trend_component = trend * t
        seasonality_component = seasonality * np.sin(2 * np.pi * t / 365)
        noise_component = noise * np.random.normal(0, 1, periods)
        
        # Combine components
        data = {
            'date': dates
        }
        
        for metric in metrics:
            data[metric] = trend_component + seasonality_component + noise_component
            
        return pd.DataFrame(data)
    
    @staticmethod
    def generate_related_tables() -> Dict[str, pd.DataFrame]:
        """Generate a set of related tables (e.g., orders, customers, products)."""
        # Generate customers
        customers = pd.DataFrame({
            'customer_id': range(1, 101),
            'name': [f'Customer_{i}' for i in range(1, 101)],
            'segment': np.random.choice(['Retail', 'Wholesale', 'Online'], 100)
        })
        
        # Generate products
        products = pd.DataFrame({
            'product_id': range(1, 51),
            'name': [f'Product_{i}' for i in range(1, 51)],
            'category': np.random.choice(['A', 'B', 'C'], 50),
            'price': np.random.uniform(10, 100, 50)
        })
        
        # Generate orders
        num_orders = 1000
        orders = pd.DataFrame({
            'order_id': range(1, num_orders + 1),
            'customer_id': np.random.choice(customers['customer_id'], num_orders),
            'product_id': np.random.choice(products['product_id'], num_orders),
            'quantity': np.random.randint(1, 10, num_orders),
            'order_date': pd.date_range(start='2023-01-01', periods=num_orders)
        })
        
        # Add order details
        orders = orders.merge(products[['product_id', 'price']], on='product_id')
        orders['total_amount'] = orders['quantity'] * orders['price']
        
        return {
            'customers': customers,
            'products': products,
            'orders': orders
        }

if __name__ == "__main__":
    # Example usage
    generator = TestDataGenerator()
    
    # Generate different types of test data
    sales_data = generator.generate_sales_data()
    inventory_data = generator.generate_inventory_data()
    customer_data = generator.generate_customer_data()
    time_series_data = generator.generate_time_series_data()
    related_tables = generator.generate_related_tables()
    
    print("Generated test data samples:") 