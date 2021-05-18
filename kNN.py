import math
import numpy as np  
import operator  
import time
import util
import random
PRINT = True

class kNNClassifier:
  """
  Perceptron classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__( self, legalLabels, num_neighbors, data_type):
    print("###########################################")
    print(legalLabels)
    self.legalLabels = legalLabels
    self.num_neighbors = num_neighbors
    self.type = "kNN"
    self.weights = {}
    for label in legalLabels:
      self.weights[label] = util.Counter() # this is the data-structure you should use
    print(self.weights)
    self.data_type = data_type

  def setWeights(self, weights):
    assert len(weights) == len(self.legalLabels)
    self.weights == weights
      
  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    The training phase of kNN consists only of storing the feature vectors and class labels of the training samples.
        
    Use the provided self.weights[label] data structure so that 
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    (and thus represents a vector a values).
    """
    self.features = trainingData[0].keys() # could be useful later
    # print(self.features)
    self.numSamples = len(trainingData)
    if self.data_type =="digits":
        self.x_train = np.ndarray(shape=(self.numSamples,28,28), dtype=float)
        self.y_train = np.ndarray(shape=(self.numSamples), dtype=int)
        for i in range(len(trainingData)):
            for idx, (k, v) in enumerate(trainingData[i].items()):
                self.x_train[i][k[0]][k[1]] = v
        for i in range(len(trainingLabels)):
            self.y_train[i] = trainingLabels[i]

    elif self.data_type == "faces":
        self.x_train = np.ndarray(shape=(self.numSamples,60,70), dtype=float)
        self.y_train = np.ndarray(shape=(self.numSamples), dtype=int)
        for i in range(len(trainingData)):
            for idx, (k, v) in enumerate(trainingData[i].items()):
                self.x_train[i][k[0]][k[1]] = v
        for i in range(len(trainingLabels)):
            self.y_train[i] = trainingLabels[i]    


  def classify(self, data):
    """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.
    
    Recall that a datum is a util.counter... 
    """
    if self.data_type =="digits":
        self.x_test = np.ndarray(shape=(len(data),28,28), dtype=float)
        for i in range(len(data)):
            for idx, (k, v) in enumerate(data[i].items()):
                self.x_test[i][k[0]][k[1]] = v
    elif self.data_type =="faces":
        self.x_test = np.ndarray(shape=(len(data),60,70), dtype=float)
        for i in range(len(data)):
            for idx, (k, v) in enumerate(data[i].items()):
                self.x_test[i][k[0]][k[1]] = v
    newInput = self.x_test        
    guesses = []
    labels = self.y_train
    ## step 1: calculate Euclidean distance  
    # tile(A, reps): Construct an array by repeating A reps times  
    # the following copy numSamples rows for dataSet
    for i in range(len(newInput)):
        diff = np.tile(newInput[i:i+1],(self.numSamples,1,1)) - self.x_train # Subtract element-wise  
        squaredDiff = diff ** 2 # squared for the subtract
        if self.data_type =="digits":
            squaredDiff = squaredDiff.reshape((self.numSamples,28*28))  
        elif self.data_type =="faces":
            squaredDiff = squaredDiff.reshape((self.numSamples,70*60))  
        distance = np.sum(squaredDiff, axis = 1) # sum is performed by row  
        ## step 2: sort the distance  
        sortedDistIndices = np.argsort(distance)
        classCount = {} # define a dictionary (can be append element)  
        for i in range(self.num_neighbors):  
            ## step 3: choose the min k distance  
            voteLabel = labels[sortedDistIndices[i]]  
            ## step 4: count the times labels occur  
            # when the key voteLabel is not in dictionary classCount, get()  
            # will return 0  
            classCount[voteLabel] = classCount.get(voteLabel, 0) + 1  
            if voteLabel in classCount:
                classCount[voteLabel] = classCount[voteLabel] + 1
            else:
                classCount[voteLabel] = 1
            ## step 5: the max voted class will return  
        maxCount = 0  
        for key, value in classCount.items():  
            if value > maxCount:  
                maxCount = value  
                maxIndex = key
        guesses.append(maxIndex)
    return guesses
    
    
    # guesses = []
    # for datum in data:
      # vectors = util.Counter()
      # for l in self.legalLabels:
        # vectors[l] = self.weights[l] * datum
      # guesses.append(vectors.argMax())
    # return guesses

  
  def findHighWeightFeatures(self, label):
    """
    Returns a list of the 100 features with the greatest weight for some label
    """
    featuresWeights = []

    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

    return featuresWeights