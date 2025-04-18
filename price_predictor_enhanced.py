import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
import os

def load_data(filepath):
    # Load the dataset
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    # Drop columns that are not useful for prediction
    data = data.drop(['id', 'date'], axis=1, errors='ignore')
    # Handle missing values if any (drop for simplicity)
    data = data.dropna()
    return data

def split_data(data):
    # Separate features and target variable
    X = data.drop('price', axis=1)
    y = data['price']
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    # Initialize and train the Random Forest Regressor
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_model_with_gridsearch(X_train, y_train):
    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    print(f"Best parameters found: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    return best_model

def evaluate_model(model, X_test, y_test):
    # Predict on test data
    y_pred = model.predict(X_test)
    # Calculate RMSE and R^2
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R^2 Score: {r2:.2f}")

def save_model(model, filepath):
    # Save the trained model to a file
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    # Load a trained model from a file
    model = joblib.load(filepath)
    return model

def predict_price(model, input_data):
    # Predict house price for new input data (input_data should be a DataFrame)
    prediction = model.predict(input_data)
    return prediction

def example_predict_new_data(model):
    # Example usage of predict_price with new input data
    import pandas as pd
    # Example input data with feature columns matching training data
    example_data = pd.DataFrame({
        'bedrooms': [3],
        'bathrooms': [2],
        'sqft_living': [1800],
        'sqft_lot': [5000],
        'floors': [1],
        'waterfront': [0],
        'view': [0],
        'condition': [3],
        'grade': [7],
        'sqft_above': [1800],
        'sqft_basement': [0],
        'yr_built': [1990],
        'yr_renovated': [0],
        'zipcode': [98103],
        'lat': [47.65],
        'long': [-122.35],
        'sqft_living15': [1800],
        'sqft_lot15': [5000]
    })
    predicted_price = predict_price(model, example_data)
    print(f"Predicted price for example input: ${predicted_price[0]:,.2f}")

def main():
    filepath = 'archive (1)/kc_house_data.csv'
    model_filepath = 'models/price_model.pkl'

    data = load_data(filepath)
    data = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(data)

    # Use hyperparameter tuning for training
    model = train_model_with_gridsearch(X_train, y_train)

    evaluate_model(model, X_test, y_test)
    save_model(model, model_filepath)

    # Example prediction with new data
    example_predict_new_data(model)

if __name__ == '__main__':
    main()
