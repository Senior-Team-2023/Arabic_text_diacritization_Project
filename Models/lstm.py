from keras.models import Sequential
from keras.layers import Dense, LSTM

class LSTM_Model:
    def __init__(self, input_shape, output_shape):
        self.model = Sequential()
        self.model.add(LSTM(units=64, input_shape=input_shape))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(output_shape, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        return loss, accuracy 
    
