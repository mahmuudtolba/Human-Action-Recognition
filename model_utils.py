from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
import tensorflow as tf

def create_model(): 
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)


    model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)),
    LSTM(128, return_sequences=True, activation='relu'),
    LSTM(64, return_sequences=False, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')])
    adam = tf.keras.optimizers.Adam(
        learning_rate=0.0001, # that small learning rate is very important
        beta_1=0.9,
        beta_2=0.999)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model