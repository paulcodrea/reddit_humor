from keras.models import Sequential
from keras.layers.core import Dense, Dropout 
from keras.layers import LSTM, Embedding

def LSTM(x_input_dim, x_output_dim, x_input_length, x_train, y_train, config):
    """
    Creates a LSTM model
    :param x_input_dim: input dimension
    :param x_output_dim: output dimension
    :param x_input_length: input length
    :param x_train: training data
    :param y_train: training labels
    :param config: configuration
    :return: model
    """
    model = Sequential()
    model.add(Embedding(input_dim=x_input_dim, output_dim=x_output_dim, input_length=int(x_input_length)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(10))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train, epochs=config['epochs'], batch_size=config['batch_size'], verbose='auto', validation_split=config['val_p'])

    return model