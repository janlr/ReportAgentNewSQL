import streamlit as st

# Access database credentials
mysql_creds = st.secrets["mysql"]
pg_creds = st.secrets["postgresql"]

# Use them in your database connections
# Example with MySQL
mysql_url = f"mysql+pymysql://{mysql_creds.user}:{mysql_creds.password}@{mysql_creds.host}:{mysql_creds.port}/{mysql_creds.database}" 