from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras import Input



class LSTM_Model:
    def __init__(self, input_shape, output_shape, num_lstm_layers = 5, hidden_size = 128):
        self.model = Sequential()
        
        self.model.add(Input(shape=input_shape))

        # Add the first LSTM layer
        self.model.add(LSTM(units=hidden_size, return_sequences=True))
        
        # Add additional LSTM layers
        for _ in range(num_lstm_layers - 1):
            self.model.add(LSTM(units=hidden_size, return_sequences=True))

        # Add the last LSTM layer without return_sequences
        self.model.add(LSTM(units=hidden_size))
        
        # # Add a Dense layer after the last LSTM layer
        # self.model.add(Dense(32, activation='relu'))
        
        # Output layer
        self.model.add(Dense(output_shape, activation='softmax'))
        
        # Compile the model
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def summary(self):
        self.model.summary()

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        return loss, accuracy
