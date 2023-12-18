# # example of a multi-label classification task
# from sklearn.datasets import make_multilabel_classification
# # define dataset
# X, Y = make_multilabel_classification(n_samples=1000, n_features=10, n_classes=3, n_labels=2, random_state=1)
# # summarize dataset shape
# print(X.shape, Y.shape)
# # summarize first few examples
# for i in range(10):
# 	print(X[i], Y[i])