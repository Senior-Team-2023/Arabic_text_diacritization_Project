from hmmlearn import hmm
import numpy as np

class HMM:
    def __init__(self, n_components, covariance_type='full'):
        self.model = hmm.GaussianHMM(n_components=n_components, covariance_type=covariance_type)
        self.trained = False

    def train(self, X_train):
        # X_train should be a 2D array where each row is a sequence of observations
        lengths = [len(seq) for seq in X_train]
        X_concatenated = np.concatenate(X_train)
        self.model.fit(X_concatenated, lengths=lengths)
        self.trained = True

    def predict(self, X_test):
        if not self.trained:
            raise Exception("Model not trained. Call train() before predict()")

        return [self.model.predict(observation.reshape(-1, 1)) for observation in X_test]

    def evaluate(self, X_test):
        if not self.trained:
            raise Exception("Model not trained. Call train() before evaluate()")

        return [self.model.score(observation.reshape(-1, 1)) for observation in X_test]
