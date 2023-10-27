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

It will return the loss and the acuracy.
##### Prediction
The model can now be used to make predictions

    print(model.predict(features))

The predict method will return a 1 if the features of a song are indicative of one that will be a hit song
otherwise a 0 if it is a flop.

### Required Features
Taken directly from the author of the [Spotify Hit Predictor Dataset (1960-2019](https://www.kaggle.com/datasets/theoverman/the-spotify-hit-predictor-dataset/), the prediction parameter must include these features, in this order, and created in this sense

        - danceability: Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable. 
        
        - energy: Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy. 
        
        - key: The estimated overall key of the track. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C?/D?, 2 = D, and so on. If no key was detected, the value is -1.
        
        - loudness: The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db. 
        
        - mode: Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.
        
        - speechiness: Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks. 
        
        - acousticness: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic. The distribution of values for this feature look like this:
        
        - instrumentalness: Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0. The distribution of values for this feature look like this:
        
        - liveness: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.
        
        - valence: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).
        
        - tempo: The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration. 
        
        - duration_ms:  The duration of the track in milliseconds.
        
        - time_signature: An estimated overall time signature of a track. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure).
        
        - chorus_hit: This the the author's best estimate of when the chorus would start for the track. Its the timestamp of the start of the third section of the track. This feature was extracted from the data received by the API call for Audio Analysis of that particular track.
        
        - sections: The number of sections the particular track has. This feature was extracted from the data received by the API call for Audio Analysis of that particular track.
