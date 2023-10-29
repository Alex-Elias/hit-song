import tkinter as tk
from tkinter import ttk
import pandas as pd
import keras
from model import Model
from model import Data_preprocessing
import joblib
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


class MachineLearningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Machine Learning Predictor")
        
        # Load your pre-trained machine learning model
        self.model = load_model("model.keras")  # Replace "model.keras" with your model's filename
 # Replace with the path to your model file

        # Create a MinMaxScaler instance
        self.scaler = joblib.load("scaler.sav")


        # Create and configure input frame
        self.input_frame = ttk.LabelFrame(root, text="Input Data")
        self.input_frame.grid(column=0, row=0, padx=10, pady=10, sticky="w")

        # Feature 1 input
        self.label_feature1 = ttk.Label(self.input_frame, text="Feature 1:")
        self.label_feature1.grid(column=0, row=0)
        self.entry_feature1 = ttk.Entry(self.input_frame)
        self.entry_feature1.grid(column=1, row=0)

        # Create and configure output frame
        self.output_frame = ttk.LabelFrame(root, text="Prediction")
        self.output_frame.grid(column=0, row=1, padx=10, pady=10, sticky="w")

        # Result label
        self.label_result = ttk.Label(self.output_frame, text="")
        self.label_result.grid(column=0, row=0)

        # Predict button
        self.predict_button = ttk.Button(root, text="Predict", command=self.predict)
        self.predict_button.grid(column=0, row=2, padx=10, pady=10)

    def predict(self):
        try:
            feature1 = float(self.entry_feature1.get())
            input_data = pd.DataFrame({"feature1": [feature1]})  # Create a DataFrame with user input

            # Scale the input data using the saved scaler
            input_data = pd.DataFrame(self.scaler.transform(input_data), columns=input_data.columns)

            # Use the machine learning model to make predictions
            prediction = self.model.predict(input_data)

            self.label_result.config(text=f"Prediction: {prediction[0][0]:.2f}")
        except ValueError:
            self.label_result.config(text="Invalid input. Please enter a valid number.")

if __name__ == "__main__":
    root = tk.Tk()
    app = MachineLearningApp(root)
    root.mainloop()
