import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load the generated dataset
df = pd.read_csv('student_data.csv')
X = df.drop(columns=['Final_Marks'])
y = df['Final_Marks']

# Split train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models benchmark suite
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, random_state=42)
}

best_model_name = None
best_r2 = -float('inf')
best_pipeline = None

print(f"{'Model':<22} | {'MAE':<8} | {'RMSE':<8} | {'R2 Score':<8}")
print("-" * 52)

for name, model in models.items():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"{name:<22} | {mae:<8.3f} | {rmse:<8.3f} | {r2:<8.3f}")
    
    if r2 > best_r2:
        best_r2 = r2
        best_model_name = name
        best_pipeline = pipeline

# Save optimal pipeline
joblib.dump(best_pipeline, 'best_student_model.pkl')
print("-" * 52)
print(f"Optimal Model Selected: {best_model_name} (R2: {best_r2:.4f})")
print("Saved serialized model to 'best_student_model.pkl'")