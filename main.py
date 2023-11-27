import utils

# Read data from train.txt and filter it from unwanted patterns
training_set = utils.read_data("train")
print(training_set[0:100])
tokenized_training_set = utils.char_tokenizer(training_set)
print(tokenized_training_set[0:10])
# Extract Features from training set

# Build Model
