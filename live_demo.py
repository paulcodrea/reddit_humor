## THIS iS A PYTHON SCRIPT TO RUN THE CODE FOR LIVE DEMO
## python C:\Users\paulc\OneDrive\Desktop\NLU_humour\reddit_humor\live_demo.py
import os
import pickle
import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize

from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

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
        input = word_tokenize(text)
        input = [word for word in input if word.isalpha()]

        print("Clean input: ", input)

        input_numerical = self.tokenizer.texts_to_sequences(input) # Convert to numerical
        input_numerical = np.array(input_numerical).reshape(1, len(input_numerical)) # 1 row, len() columns
        input_numerical = pad_sequences(input_numerical, maxlen=int(self.max_length), padding='post') # Pad the input
        
        self.input_vec = input_numerical # Set the input vector

    def run_model(self):
        """
        Runs the model and returns the output
        """
        output = self.model.predict(self.input_vec)

        if int(output) == 0:
            print("The input is not a joke")
        else:
            print("The input is a joke")

########################################################################################################################
# THIS SHOULD BE UPDATED TO USE THE NEW MODEL
path = 'C:\\Users\\paulc\\OneDrive\\Desktop\\NLU_humour\\reddit_humor\\model\\'
live_session = humour_live_demo(path)


for file in os.listdir(live_session.path):
    if file.endswith(".h5"):
        model_name = file[:2] # parse file name to get first 2 characters from file name

        # 1. Read Model
        try: 
            live_session.model = load_model(live_session.path + file)
            print("\n______________________________________________________________")
            # print("\nLoading: " + model_name)
            print("Model loaded: " + file)
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

        ################ READ INPUT FROM USER #############################
        live_session.max_length = live_session.data_df['max_len'][0]
        print("Max length: ", live_session.max_length)
        
        ret = input("Enter a sentence: ")
        live_session.process_input(ret)
        live_session.run_model()
        ###################################################################
        print("______________________________________________________________")

print("End of model...")