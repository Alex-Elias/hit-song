import tkinter as tk
from tkinter import ttk
import pandas as pd
import keras
from model import Model
from model import Data_preprocessing
import joblib
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk
from tkinter import ttk
import keras
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class MachineLearningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Machine Learning Predictor")

        # Load your pre-trained machine learning model

        self.model = load_model("model.keras")  # Replace with the path to your model file

        # Create and configure input frame
        self.input_frame = ttk.LabelFrame(root, text="Input Data")
        self.input_frame.grid(column=0, row=0, padx=10, pady=10, sticky="w")

        # Create an empty DataFrame to store user inputs
        self.user_input_data = pd.DataFrame()

        labels_list = ['danceability (0-1)', 'energy (0-1)', 'key (0-10)', 
                       'loudness (-40+)', 'mode (0 or 1)', 'speechiness (0-1)', 
                       'acousticness (0-1)', 'instrumentalness (0-1)', 
                       'liveness (0-1)', 'valence (0-1)', 'tempo (0-500)', 
                       'duration_ms (0+)', 'time_signature (0-16)', 
                       'chorus_hit (0+)', 'sections (0+)']

        # Create input labels and entry fields for 15 features
        self.feature_entries = []
        for i in range(15):
            label = ttk.Label(self.input_frame, text=f"Input your estimated value for {labels_list[i]} of your song:")
            label.grid(column=0, row=i)
            entry = ttk.Entry(self.input_frame)
            entry.grid(column=1, row=i)
            self.feature_entries.append(entry)

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
            # Collect user input values and create a DataFrame
            input_values = []
            for entry in self.feature_entries:
                value = entry.get().strip()
                if value:
                    input_values.append(float(value))
                else:
                    raise ValueError("Please enter values for all features.")

            if len(input_values) != 15:
                raise ValueError("Please enter values for all 15 features.")

            input_data = pd.DataFrame(
                [input_values], columns=[f"Feature {i + 1}" for i in range(15)]
            )

            # Concatenate user input data with existing data
            self.user_input_data = pd.concat(
                [self.user_input_data, input_data], ignore_index=True
            )

            # Create a MinMaxScaler and fit it to the combined data
            scaler = MinMaxScaler()
            combined_data = pd.concat([self.user_input_data], ignore_index=True)
            scaler.fit(combined_data)

            # Scale the input data
            input_data = pd.DataFrame(
                scaler.transform(input_data), columns=input_data.columns
            )

            # Use the machine learning model to make predictions
            prediction = self.model.predict(input_data)

            if round(prediction[0][0], 2) > 0:
                self.label_result.config(text=f"Congratz! This song would likely be a hit song!")
            else:
                self.label_result.config(text="So sorry:(( This song would likely not be a hit song")
            
        except ValueError as e:
            self.label_result.config(text=str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = MachineLearningApp(root)
    root.mainloop()