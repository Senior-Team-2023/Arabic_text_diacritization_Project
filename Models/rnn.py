import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras import Input

class RNN_Model:
    def __init__(self, input_shape, output_shape, num_lstm_layers = 5, hidden_size = 128):
        self.model = Sequential()
        
        self.model.add(Input(shape=input_shape))

        # Add the first LSTM layer
        self.model.add(SimpleRNN(units=hidden_size, return_sequences=True))
        
        # Add additional LSTM layers
        for _ in range(num_lstm_layers - 1):
            self.model.add(SimpleRNN(units=hidden_size, return_sequences=True))

        # Add the last LSTM layer without return_sequences
        self.model.add(SimpleRNN(units=hidden_size))
        
        # # Add a Dense layer after the last LSTM layer
        # self.model.add(Dense(32, activation='relu'))
        
        # Output layer
        self.model.add(Dense(output_shape, activation='softmax'))
        
        # Compile the model
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def summary(self):
        self.model.summary()

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size = batch_size)

    def train_batch(self, padded_sequences, y_train, epochs=10, batch_size=32):
        # Train the model on each line separately
        for epoch in range(epochs):
            for i in range(len(padded_sequences)):
                input_sequence = padded_sequences[i]
                target_label = y_train[i]
                self.model.train_on_batch(tf.expand_dims(input_sequence, axis=0), tf.expand_dims(target_label, axis=0))
                print(f"Epoch: {epoch+1}/{epochs}, Line: {i+1}/{len(padded_sequences)}", end="\r")

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        return loss, accuracy

# import torch
# import torch.nn as nn
# import torch.optim as optim

# class RNN_Model(nn.Module):
#     def __init__(self, input_shape, output_shape, hidden_size):
#         super(RNN_Model, self).__init__()

#         self.hidden_size = hidden_size

#         # Define the RNN layer
#         self.rnn = nn.RNN(input_shape, hidden_size, batch_first=True)

#         # Define the fully connected layer
#         self.fc = nn.Linear(hidden_size, output_shape)

#         # Define the softmax activation
#         self.softmax = nn.LogSoftmax(dim=2)

#     def forward(self, input_seq):
#         # Forward pass through RNN
#         rnn_out, _ = self.rnn(input_seq)

#         # Get the output from the last time step
#         rnn_out_last = rnn_out[:, -1]

#         # Fully connected layer
#         output = self.fc(rnn_out_last)

#         # Apply softmax activation
#         output = self.softmax(output)

#         return output

#     def train_model(self,  X_train, y_train, epochs=100, learning_rate=0.001):
#         X_train = torch.from_numpy(X_train).float()
#         y_train = torch.from_numpy(y_train).long()
#         criterion = nn.NLLLoss()
#         optimizer = optim.Adam(self.parameters(), lr=learning_rate)

#         for epoch in range(epochs):
#             self.train()
#             optimizer.zero_grad()
#             output = self(X_train)
#             loss = criterion(output.squeeze(), y_train.squeeze())
#             loss.backward()
#             optimizer.step()

#             if (epoch + 1) % 10 == 0:
#                 print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

#     def predict(self, input_seq):
#         self.eval()
#         with torch.no_grad():
#             output = self(input_seq)
#         return output.argmax(dim=2)
