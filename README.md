# House Price Predictor

## Project Description
This project is a house price prediction system that uses a Random Forest Regressor model to predict house prices based on various features such as number of bedrooms, bathrooms, square footage, location, and more. The project includes scripts for training the model with hyperparameter tuning and a graphical user interface (GUI) application for making predictions using the trained model.

## Features
- Data loading and preprocessing
- Model training with hyperparameter tuning using GridSearchCV
- Model evaluation with RMSE and R² metrics
- Save and load trained model
- Tkinter-based GUI for interactive price prediction
- Example prediction with new input data

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Required Python Packages
Install the required packages using pip:

```bash
pip install pandas scikit-learn numpy joblib tkinter
```

Note: `tkinter` is usually included with standard Python installations. If not, install it separately based on your OS.

## Usage

### Training the Model
To train the model, run the `price_predictor_enhanced.py` script. This will load the dataset, preprocess the data, perform hyperparameter tuning, evaluate the model, and save the trained model to `models/price_model.pkl`.

```bash
python price_predictor_enhanced.py
```

Make sure the dataset file `kc_house_data.csv` is located in the `archive (1)` directory.

### Running the GUI Application
After training the model, you can run the GUI application to predict house prices interactively.

```bash
python price_predictor_gui.py
```

The GUI will load the saved model and provide input fields for all features. Enter the desired values and click "Predict Price" to see the predicted house price.

## Model Details
- Model Type: Random Forest Regressor
- Hyperparameter tuning performed using GridSearchCV
- Features used include bedrooms, bathrooms, square footage, floors, waterfront, view, condition, grade, year built, zipcode, latitude, longitude, and more.

## File Structure
```
.
├── price_predictor_enhanced.py    # Script for training and evaluating the model
├── price_predictor_gui.py         # Tkinter GUI application for price prediction
├── models/
│   └── price_model.pkl            # Saved trained model
├── archive (1)/
│   └── kc_house_data.csv          # Dataset file
└── README.md                     # This file
```

## License
This project is provided as-is without any warranty. Feel free to use and modify it for your own purposes.
