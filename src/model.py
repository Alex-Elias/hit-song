import numpy as np
import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
import sklearn
from sklearn.preprocessing import MinMaxScaler
import joblib


class Data_preprocessing:
    """
    The class to preprocess the data for the Model class
    
    Args:
        data (pandas Dataframe): The dataframe containing all data. Must all be numerical
        target (str): The name of the column containing the target variable 
    
    Attributes:
        data (pandas Dataframe): the processed data with the target axes removed
        target (pandas Dataframe): The target values
        scaler (MinMaxScaler): trained MinMaxScaler
    """
    def __init__(self, data: pd.DataFrame, target: str) -> None:
        self.data = data.drop(target, axis=1)
        self.target = data[target]
        self.__scale()
    
    def __scale(self) -> None:
        scaler = MinMaxScaler()
        scaler.fit(self.data)
        self.scaler = scaler

        self.data = scaler.transform(self.data)

    def split(self, size: float) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Splits the data into training and testing data

        Args:
            size (float): The percentage of testing to training data in range [0...1]

        Returns:
            (np.ndarray, np.ndarray, np.ndarray, np.ndarray): a tuple of four elements containing
            (training parameter, testing parameters, training target, testing target) 
        """
        return train_test_split(self.data, self.target, test_size=size, random_state=0)
    
    def get_scaler(self) -> MinMaxScaler:
        """
        Returns the trained MinMaxScaler object

        Returns:
            (MinMaxScaler): the trained MinMaxScaler object
        """
        return self.scaler
    


class Model:
    """
    The Model class contains all the functions needed to train and test a machine learning model.
    The built-in create_model method creates a simple feed-forward artifical neural network. 
    Contains the set_model method to be able to set your own model

    Attributes:
        opt (keras.optimizers.Adam): stores the Adam optimizer
        model (Sequential): stores the sequential model
        scaler (MinMaxScaler): trained MinMaxScaler
    """
    opt = keras.optimizers.Adam(learning_rate=0.001)
    scaler = None

    def create_model(self, input: int) -> None:
        """
        The create_model function creates the built-in feed-forward artifical neural network model.
        The architecture of the model is as follows: it is fully connected with an input layer consisting
        of 15 nodes, there is one hidden layer with 30 nodes. Both the mentioned layer use the relu function
        as the activation function. The output layer consists of one node with the sigmoid function.

        Args:
            input (int): The dimension of the input i.e. how many parameters does it consist of.
        """
        seq_model = Sequential()
        seq_model.add(Dense(units=15, activation="relu", input_shape=(input,)))

        seq_model.add(Dense(30, activation="relu"))

        seq_model.add(Dense(1, activation="sigmoid"))
        
        seq_model.compile(loss="binary_crossentropy", optimizer=self.opt, metrics=["accuracy"])

        self.model = seq_model

    def set_model(self, model: Sequential) -> None:
        """
        The set_model method sets the a user generated model instead of the default model

        Args:
            model (Sequential): The model to replace the default model
        """
        self.model = model
    
    def set_optimizer(self, optimizer: keras.optimizers) -> None:
        """
        The set_optimizer method sets a user specified optimizer

        Args:
            optimizer (keras.optimizer): The optimizer to replace the default
        """
        self.opt = optimizer

    def train(self, parameters: pd.DataFrame, target: pd.DataFrame, epochs: int) -> None:
        """
        The train method trains the model with the given training dataset

        Args:
            parameters (pd.DataFrame): The training parameters
            target (pd.DataFrame): The target parameter
            epochs (int): The number of epochs 
        """
        self.model.fit(parameters, target, epochs=epochs, verbose=5)
    
    def evaluate(self, t_parameters: pd.DataFrame, t_target: pd.DataFrame) -> list:
        """
        The evaluate method evaluates the model with the given test dataset

        Args:
            t_parameters (pd.DataFrame): The testing dataset consisting of the parameters
            t_target (pd.DataFrame): The testing target dataset
        
        Returns:
            [list]: Returns a list of the loss and accuracy of the model with the given testing dataset
        """
        loss_and_metrics = self.model.evaluate(t_parameters, t_target)
        return loss_and_metrics
    
    def predict(self, parameters: pd.DataFrame) -> int:
        """
        The predict method uses the binary classifier model to predict the class of the object.

        Args:
            parameters (pd.DataFrame): The parameters used to predict the class. They must be the same 
            and in the same order as the training data
        
        Returns:
            (int): Returns either a 1 or 0
        """

        parameters = self.scaler.transform(parameters)
        predicted = self.model.predict(parameters)
        return predicted
    
    def set_scaler(self, scaler: MinMaxScaler) -> None:
        """
        Sets the MinMaxScaler used in the predict method

        Args:
            scaler (MinMaxScaler): the trained MinMaxScaler used to scale the features for the predict method
        """
        self.scaler = scaler
    

def main():
    df = pd.read_csv("./data/dataset-of-10s.csv")
    
    df.drop(["track", "artist", "uri"], axis=1, inplace=True)

    dp = Data_preprocessing(df, "target")
    x_train, x_test, y_train, y_test = dp.split(0.1)

    scaler = dp.get_scaler()

    model = Model()
    model.create_model(15)
    joblib.dump(scaler, open("scalerrrr.sav", "wb"))

    model.set_scaler(scaler)

    model.train(x_train, y_train, 500)


    model.model.save("./model.keras")

    print(model.evaluate(x_test, y_test))

    

if __name__ == "__main__":
    main()
