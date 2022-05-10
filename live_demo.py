## THIS iS A PYTHON SCRIPT TO RUN THE CODE FOR LIVE DEMO
## python C:\Users\paulc\OneDrive\Desktop\NLU_humour\reddit_humor\live_demo.py
import os
import re
import pickle
import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from sklearn.ensemble import RandomForestClassifier

class humour_live_demo:
    def __init__(self, path):
        self.path = path
        self.tokenizer = None
        self.model = None
        self.data_df = pd.DataFrame()
        
        self.max_length = None
        self.input_vec = []


    def process_input(self, text):
        """
        Processes the input and returns the output
        """
        text = re.sub(r'http\S+', '', text) # remove links
        text = re.sub(r'[^\w\s]','', text) # remove punctuation

        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+", flags=re.UNICODE)

        text = emoji_pattern.sub(r'', text) # remove emoji
        text = ' '.join([word for word in word_tokenize(text) if word not in stopwords]) # remove stopwords
        text = text.lower()
        
        input = word_tokenize(text)
        input = [word for word in input if word.isalpha()]
        input = [word for word in input if not word.startswith("http")]
        print("Clean input: ", ' '.join(input))

        input_numerical = self.tokenizer.texts_to_sequences(input) # Convert to numerical
        input_numerical = [item for sublist in input_numerical for item in sublist] # Flatten the list
        input_numerical = np.array([input_numerical]) # Convert to numpy array
        input_numerical = pad_sequences(input_numerical, maxlen=int(self.max_length)) #, padding='post') # Pad the input
        input_numerical = np.array(input_numerical, dtype=np.float32)
        
        self.input_vec = input_numerical # Set the input vector

    def run_model(self):
        """
        Runs the model and returns the output
        """
        output = self.model.predict(self.input_vec)
        output = np.round(output)
        print("\nModel predicts: ", int(output))

        if int(output) == 0:
            print("The input is predicted as - not a joke")
        else:
            print("The input is predicted as - a joke")

########################################################################################################################
# THIS SHOULD BE UPDATED TO USE THE NEW MODEL
path = 'C:\\Users\\paulc\\OneDrive\\Desktop\\NLU_humour\\reddit_humor\\model\\'
live_session = humour_live_demo(path)

# print("______________________________________________________________")
# Strart the model here. Read input from user one line at a time.
while True:
    ret = input("Enter a sentence: ") # Read input from user

    # Check if the user wants to exit
    if ret == "exit":
        break
    print("\n______________________________________________________________")
    # Process the input for each of the models.
    for file in os.listdir(live_session.path):
        if file.endswith(".h5"):
            model_name = file[:2] # parse file name to get first 2 characters from file name

            # 1. Read Model
            try: 
                live_session.model = load_model(live_session.path + file)
                print("\n\nModel loaded: " + file)
            except Exception as e:
                print(e)
                print("Model not loaded")
                continue

            # 2. Read Data 
            try:
                live_session.data_df = pd.read_csv(live_session.path + model_name + '_data.csv')
                print("Data loaded: " + model_name + '_data.csv')
            except:
                print("No data found")
                continue

            # 3. Read Pickle file
            try:
                with open(live_session.path + model_name + '_tokenizer.pickle', 'rb') as handle:
                    live_session.tokenizer = pickle.load(handle)
                print("Tokenizer loaded: " + model_name + '_tokenizer.pickle')
            except:
                print("No tokenizer found")
                continue
            
            # Save Max Length
            live_session.max_length = int(live_session.data_df['max_len'][0])

            # 4. Process the input
            live_session.process_input(ret)
            live_session.run_model()

        # THIS IS FOR RANDOM FOREST ---------------------------------------------------------------------------------
        elif file.startswith("3_model"):
            model_name = file[:1]
            # 1. Read Model
            try: 
                with open(live_session.path + model_name + '_model.pickle', 'rb') as handle:
                    live_session.model = pickle.load(handle)
                print("\n\nModel loaded: " + file)
            except Exception as e:
                print(e)
                print("Model not loaded")
                continue

            # 2. Read Data 
            try:
                live_session.data_df = pd.read_csv(live_session.path + model_name + '_data.csv')
                print("Data loaded: " + model_name + '_data.csv')
            except:
                print("No data found")
                continue

            # 3. Read Pickle file
            try:
                with open(live_session.path + model_name + '_tokenizer.pickle', 'rb') as handle:
                    live_session.tokenizer = pickle.load(handle)
                print("Tokenizer loaded: " + model_name + '_tokenizer.pickle')
            except:
                print("No tokenizer found")
                continue
            
            # Save Max Length
            live_session.max_length = int(live_session.data_df['max_len'][0])

            # 4. Process the input
            live_session.process_input(ret)
            live_session.run_model()

    print("______________________________________________________________")
    print("\n")


print("End of model...")