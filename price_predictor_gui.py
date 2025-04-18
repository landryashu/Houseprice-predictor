import tkinter as tk
from tkinter import messagebox
import pandas as pd
import joblib
import os

MODEL_PATH = 'models/price_model.pkl'

class PricePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("House Price Predictor")

        # Load model
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror("Error", f"Model file not found at {MODEL_PATH}")
            root.destroy()
            return
        self.model = joblib.load(MODEL_PATH)

        # Define feature labels and default values
        self.features = [
            ('bedrooms', 3),
            ('bathrooms', 2),
            ('sqft_living', 1800),
            ('sqft_lot', 5000),
            ('floors', 1),
            ('waterfront', 0),
            ('view', 0),
            ('condition', 3),
            ('grade', 7),
            ('sqft_above', 1800),
            ('sqft_basement', 0),
            ('yr_built', 1990),
            ('yr_renovated', 0),
            ('zipcode', 98103),
            ('lat', 47.65),
            ('long', -122.35),
            ('sqft_living15', 1800),
            ('sqft_lot15', 5000)
        ]

        self.entries = {}
        for i, (feature, default) in enumerate(self.features):
            label = tk.Label(root, text=feature)
            label.grid(row=i, column=0, padx=5, pady=5, sticky='e')
            entry = tk.Entry(root)
            entry.grid(row=i, column=1, padx=5, pady=5)
            entry.insert(0, str(default))
            self.entries[feature] = entry

        self.predict_button = tk.Button(root, text="Predict Price", command=self.predict_price)
        self.predict_button.grid(row=len(self.features), column=0, columnspan=2, pady=10)

        self.result_label = tk.Label(root, text="", font=('Arial', 14))
        self.result_label.grid(row=len(self.features)+1, column=0, columnspan=2, pady=10)

    def predict_price(self):
        try:
            input_data = {}
            for feature, _ in self.features:
                value = self.entries[feature].get()
                # Convert to appropriate type (float or int)
                if feature in ['zipcode', 'yr_built', 'yr_renovated', 'bedrooms', 'bathrooms', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_basement']:
                    input_data[feature] = int(value)
                else:
                    input_data[feature] = float(value)

            input_df = pd.DataFrame([input_data])
            prediction = self.model.predict(input_df)[0]
            self.result_label.config(text=f"Predicted Price: ${prediction:,.2f}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to predict price: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PricePredictorApp(root)
    root.mainloop()
