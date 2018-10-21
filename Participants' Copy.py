# Perceptron Algorithm on the Sonar Dataset
from random import seed
from random import randrange
from csv import reader

# Load the data & format it
def load_data(filename):
    # Read from the CSV file
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)

    '''The data imported is in the format below:
        [
            [x1_1,x1_2,x1_3, ..., x1_60, y1],
            [x2_1,x2_2,x2_3, ..., x2_60, y2],
            ...
            ]
    Each row is one set of data consisting of 61 items,
    the first 60 are the input values (i.e. the sonar data),
    the last one is the actual result of either a 'M' (metal) or 'R' (rock)'''

    # Convert the string of the input values to float
    # Use 0 & 1 to represent 'M' and 'R' respectively for calculation
    for row in dataset:
        for i in range(len(row)-1): # leave out the last one
            row[i] = float(row[i].strip())

        if row[-1] == 'M':  # metal
            row[-1] = 0
        else:   # rock
            row[-1] = 1

    return dataset


# Prediction function
def predict(row, weights):
    '''Recall:
    h(x) = w0 + w1*x0 + w2*x1 + ... (as x is 0-indexed)

    If h(x) < 0, result is 0 (metal)
    If h(x) > 0, result is 1 (rock)
    '''
    # Calculate h(x)

    # ReLU


# train weights
def train_weights(train_data, l_rate, n_epoch):
    # Initialise the weights to zero


    # Each epoch is the process of using all the training data for training once.
    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in train_data:
            

            # Count no. of wrong predictions
            

            # Update w0
            # Update w1, w2, ...

        # Show the process
        # by printing out the no. of errors after every 5 epoches
        
    return weights


# perceptron model
def perceptron(train_data, test_data, l_rate, n_epoch):
    # Train the model
    weights = train_weights(train_data, l_rate, n_epoch)

    # Make predictions
    predictions = []
    for row in test_data:
        prediction = predict(row, weights)
        predictions.append(prediction)

    return(predictions)


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = []
    dataset_copy = list(dataset) # do not change the original dataset list
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Evaluate an algorithm using a cross validation split
def evaluate(dataset, n_folds, l_rate, n_epoch):
    folds = cross_validation_split(dataset, n_folds)
    scores = []

    fold_count = 0
    for fold in folds:
        # each time take all except for one fold for training,
        # and the remaining one fold for testing
        train_data = list(folds)
        train_data.remove(fold)
        train_data = sum(train_data, [])
        
        test_data = []
        actual = []
        for row in fold:
            row_copy = list(row)
            test_data.append(row_copy)

            actual.append(row_copy[-1])
            row_copy[-1] = None


        print("Fold no.: %d" % fold_count)
        fold_count += 1

        # Train and make predictions
        predicted = perceptron(train_data, test_data, l_rate, n_epoch)

        # Evaluate
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores



# Load and Prepare data
filename = 'sonar.all-data.csv'
dataset = load_data(filename)


# Define Hyper-parameters
n_folds = 3
l_rate = 0.01
n_epoch = 100


# Evaluate the algorithm
scores = evaluate(dataset, n_folds, l_rate, n_epoch)
print('Scores: %s' % scores)
print('Average Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
