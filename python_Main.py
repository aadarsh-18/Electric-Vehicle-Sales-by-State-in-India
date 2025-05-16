# ev_sales_india_project.py

# ========== STEP 1: Import Libraries ==========
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

# ========== STEP 2: Setup Logging ==========
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ========== STEP 3: Generate or Load EV Sales Data ==========
def generate_ev_sales_data():
    logging.info("Generating synthetic EV sales data by state in India...")
    states = [
        'Maharashtra', 'Delhi', 'Karnataka', 'Tamil Nadu', 'Uttar Pradesh',
        'Gujarat', 'Rajasthan', 'Haryana', 'Telangana', 'Kerala'
    ]
    years = list(range(2018, 2024))

    rows = []
    for state in states:
        for year in years:
            ev_sales = np.random.randint(1000, 10000)
            subsidy = np.random.uniform(0.1, 0.5)  # Subsidy as % of base cost
            population = np.random.randint(1_000_000, 10_000_000)
            income_index = np.random.uniform(0.5, 1.5)
            infra_index = np.random.uniform(0.4, 1.2)
            rows.append([state, year, ev_sales, subsidy, population, income_index, infra_index])

    df = pd.DataFrame(rows, columns=[
        'State', 'Year', 'EV_Sales', 'Subsidy_Rate', 'Population', 'Income_Index', 'Infra_Index'])
    logging.info("Data generation complete.")
    return df

# ========== STEP 4: Store in SQL ==========
def store_to_sql(df, db_name='ev_sales.db'):
    logging.info("Storing EV sales data to SQLite database...")
    conn = sqlite3.connect(db_name)
    df.to_sql('ev_sales', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()
    logging.info("Data stored successfully.")

# ========== STEP 5: Load from SQL ==========
def load_from_sql(db_name='ev_sales.db'):
    logging.info("Loading EV sales data from SQLite database...")
    conn = sqlite3.connect(db_name)
    df = pd.read_sql('SELECT * FROM ev_sales', conn)
    conn.close()
    logging.info("Data loaded successfully.")
    return df

# ========== STEP 6: Exploratory Data Analysis ==========
def plot_eda(df):
    logging.info("Generating EDA visualizations...")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Year', y='EV_Sales', hue='State', marker='o')
    plt.title('EV Sales by State Over Years')
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()

# ========== STEP 7: Feature Engineering ==========
def prepare_features(df):
    logging.info("Preparing features for ML model...")
    df_encoded = pd.get_dummies(df, columns=['State'], drop_first=True)
    X = df_encoded.drop(['EV_Sales'], axis=1)
    y = df_encoded['EV_Sales']
    return X, y

# ========== STEP 8: Machine Learning Model ==========
def train_model(X, y):
    logging.info("Training Random Forest Regressor model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)

    logging.info(f"RMSE: {rmse:.2f}")
    logging.info(f"MAE: {mae:.2f}")
    return model

# ========== STEP 9: Export to Excel ==========
def export_to_excel(df, filename='EV_Sales_Report.xlsx'):
    logging.info("Exporting final dataset to Excel...")
    df.to_excel(filename, index=False)
    logging.info("Export complete.")

# ========== STEP 10: Full Pipeline ==========
def run_pipeline():
    df = generate_ev_sales_data()
    store_to_sql(df)
    df_sql = load_from_sql()
    plot_eda(df_sql)
    X, y = prepare_features(df_sql)
    model = train_model(X, y)
    export_to_excel(df_sql)

if __name__ == "__main__":
    run_pipeline()
