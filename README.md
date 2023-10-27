# Project for the introduction to Data Science course
## Hit songs
This is our project for the introduction to Data Science course, we used the [Spotify Hit Predictor Dataset (1960-2019](https://www.kaggle.com/datasets/theoverman/the-spotify-hit-predictor-dataset/) from Kaggle to train a machine learning model to predict hit songs.

### Installation of Requirements
Python 3.10 was used to run this project and requirements are stored in the requirements.txt file. They can be installed with running the following command in the terminal <code> pip3 install -r requirements.txt</code>

### Quick Start Guide
The model.py file contains the classes required to preprocess the dataset from Kaggle and train a model to predict hit songs.
#### Example code
Example code to create your own fitted model.

    df = pd.read_csv("Path_to_dataset")
    
    #drops unnecessary columns from the DataFrame
    df.drop(["track", "artist", "uri"], axis=1, inplace=True)

    #creates a data_preprocessing object with the df DataFrame and "target" as the target column name
    dp = Data_preprocessing(df, "target")

    #splits the data into training and testing with a 0.25 split
    x_train, x_test, y_train, y_test = dp.split(0.25)

    #creates a basic feed forward artificial neural network model with an input of 15 features
    model = Model()
    model.create_model(15)

    #trains the model with the features and target DataFrame with 100 epochs
    model.train(x_train, y_train, 100)


It is also possible to use our pretrained model model.keras, it can be open as in the following

    # define dpendency
    from tensorflow.keras.models import load_model

    # load model
    model = load_model("model.keras")

    # check model info
    model.summary()

##### Evaluation
The model can be evaluated in the following way

    model.evaluate(x_test, y_test)
##### Prediction
The model can now be used to make predictions

    print(model.predict(features))

The predict method will return a 1 if the features of a song are indicative of one that will be a hit song
otherwise a 0 if it is a flop.
