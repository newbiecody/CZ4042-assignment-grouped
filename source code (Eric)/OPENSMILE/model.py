from contextlib import redirect_stdout
from kerastuner.engine.hyperparameters import HyperParameters as hp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization

def sm_model (hp):
    model = Sequential()
    model.add(Conv1D(256, kernel_size=8, strides = 1, activation='relu', input_shape=(988,1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))

    model.add(Conv1D(128, kernel_size=8, strides = 1, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))

    model.add(Conv1D(64, kernel_size=8, strides = 1, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.7))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    
    opt = hp.Choice("opt", values=['adam', 'sgd', 'rmsprop', 'adadelta', 'adamax', 'adagrad'])
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])

    #with open('model_summary.txt', 'w') as f:
    #    with redirect_stdout(f):
    #        model.summary()

    return model

