# dataClassifier.py
# -----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# This file contains feature extraction methods and harness 
# code for data classification

import mostFrequent
import naiveBayes
import perceptron
import mira
import kNN
import samples
import sys
import util
import operator  
import time
import random

TRAIN_SET_SIZE_DIGITS = 5000
VALID_SET_SIZE_DIGITS = 1000
TEST_SET_SIZE_DIGITS = 1000
TRAIN_SET_SIZE_FACES = 451
VALID_SET_SIZE_FACES = 301
TEST_SET_SIZE_FACES = 150
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70


def basicFeatureExtractorDigit(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is white (0) or gray/black (1)
  """
  a = datum.getPixels()

  features = util.Counter()
  for x in range(DIGIT_DATUM_WIDTH):
    for y in range(DIGIT_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features

def basicFeatureExtractorFace(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is an edge (1) or no edge (0)
  """
  a = datum.getPixels()

  features = util.Counter()
  for x in range(FACE_DATUM_WIDTH):
    for y in range(FACE_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features

def enhancedFeatureExtractorDigit(datum):
  """
  Your feature extraction playground.
  
  You should return a util.Counter() of features
  for this datum (datum is of type samples.Datum).
  
  ## DESCRIBE YOUR ENHANCED FEATURES HERE...
  
  ##
  """
  features =  basicFeatureExtractorDigit(datum)

  "*** YOUR CODE HERE ***"
  
  return features


def contestFeatureExtractorDigit(datum):
  """
  Specify features to use for the minicontest
  """
  features =  basicFeatureExtractorDigit(datum)
  return features

def enhancedFeatureExtractorFace(datum):
  """
  Your feature extraction playground for faces.
  It is your choice to modify this.
  """
  features =  basicFeatureExtractorFace(datum)
  return features

def analysis(classifier, guesses, testLabels, testData, rawTestData, printImage):
  """
  This function is called after learning.
  Include any code that you want here to help you analyze your results.
  
  Use the printImage(<list of pixels>) function to visualize features.
  
  An example of use has been given to you.
  
  - classifier is the trained classifier
  - guesses is the list of labels predicted by your classifier on the test set
  - testLabels is the list of true labels
  - testData is the list of training datapoints (as util.Counter of features)
  - rawTestData is the list of training datapoints (as samples.Datum)
  - printImage is a method to visualize the features 
  (see its use in the odds ratio part in runClassifier method)
  
  This code won't be evaluated. It is for your own optional use
  (and you can modify the signature if you want).
  """
  
  # Put any code here...
  # Example of use:
  for i in range(len(guesses)):
      prediction = guesses[i]
      truth = testLabels[i]
      if (prediction != truth):
          print("===================================")
          print("Mistake on example %d" % i)
          print("Predicted %d; truth is %d" % (prediction, truth))
          print("Image: ")
          print(rawTestData[i])
          print("===================================")
          break


## =====================
## You don't have to modify any code below.
## =====================


class ImagePrinter:
    def __init__(self, width, height):
      self.width = width
      self.height = height

    def printImage(self, pixels):
      """
      Prints a Datum object that contains all pixels in the 
      provided list of pixels.  This will serve as a helper function
      to the analysis function you write.
      
      Pixels should take the form 
      [(2,2), (2, 3), ...] 
      where each tuple represents a pixel.
      """
      image = samples.Datum(None,self.width,self.height)
      for pix in pixels:
        try:
            # This is so that new features that you could define which 
            # which are not of the form of (x,y) will not break
            # this image printer...
            x,y = pix
            image.pixels[x][y] = 2
        except:
            print("new features:", pix)
            continue
      print(image)  

def default(str):
  return str + ' [Default: %default]'

def readCommand( argv ):
  "Processes the command used to run from the command line."
  from optparse import OptionParser  
  parser = OptionParser(USAGE_STRING)
  
  parser.add_option('-c', '--classifier', help=default('The type of classifier'), choices=['mostFrequent', 'nb', 'naiveBayes', 'perceptron', 'mira', 'minicontest', 'kNN'], default='mostFrequent')
  parser.add_option('-d', '--data', help=default('Dataset to use'), choices=['digits', 'faces'], default='digits')
  parser.add_option('-t', '--training', help=default('The size of the training set'), default=100, type="int")
  parser.add_option('-f', '--features', help=default('Whether to use enhanced features'), default=False, action="store_true")
  parser.add_option('-o', '--odds', help=default('Whether to compute odds ratios'), default=False, action="store_true")
  parser.add_option('-1', '--label1', help=default("First label in an odds ratio comparison"), default=0, type="int")
  parser.add_option('-2', '--label2', help=default("Second label in an odds ratio comparison"), default=1, type="int")
  parser.add_option('-w', '--weights', help=default('Whether to print weights'), default=False, action="store_true")
  parser.add_option('-k', '--smoothing', help=default("Smoothing parameter (ignored when using --autotune)"), type="float", default=2.0)
  parser.add_option('-a', '--autotune', help=default("Whether to automatically tune hyperparameters"), default=False, action="store_true")
  parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=3, type="int")
  parser.add_option('-s', '--testdigits', help=default("Amount of DIGITS test data to use, MAX is 1000"), default=TEST_SET_SIZE_DIGITS, type="int")
  parser.add_option('-m', '--testfaces', help=default("Amount of FACES test data to use, MAX is 150"), default=TEST_SET_SIZE_FACES, type="int")
  parser.add_option('-n', '--neighbors', help=default("Number of Neighbors"), default=5, type="int")

  options, otherjunk = parser.parse_args(argv)
  if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
  args = {}
  
  # Set up variables according to the command line input.
  print("====================================================================================================")
  print("Doing classification")
  print("Data: {0}".format(options.data))
  print("Classifier: {0}".format(options.classifier))
  if not options.classifier == 'minicontest':
    print("Using enhanced features?: {0}".format(options.features))
  else:
    print("Using minicontest feature extractor")
  print("Training set size: {0}".format(options.training))
  if(options.data=="digits"):
    printImage = ImagePrinter(DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT).printImage
    if (options.features):
      featureFunction = enhancedFeatureExtractorDigit
    else:
      featureFunction = basicFeatureExtractorDigit
    if (options.classifier == 'minicontest'):
      featureFunction = contestFeatureExtractorDigit
  elif(options.data=="faces"):
    printImage = ImagePrinter(FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT).printImage
    if (options.features):
      featureFunction = enhancedFeatureExtractorFace
    else:
      featureFunction = basicFeatureExtractorFace      
  else:
    print("Unknown dataset {0}".format(options.data))
    print(USAGE_STRING)
    sys.exit(2)
    
  if(options.data=="digits"):
    legalLabels = range(10)
  else:
    legalLabels = range(2)
    
  if options.training <= 0:
    print("Training set size should be a positive integer (you provided: %d)" % options.training)
    print(USAGE_STRING)
    sys.exit(2)
    
  if options.smoothing <= 0:
    print("Please provide a positive number for smoothing (you provided: %f)" % options.smoothing)
    print(USAGE_STRING)
    sys.exit(2)
    
  if options.odds:
    if options.label1 not in legalLabels or options.label2 not in legalLabels:
      print("Didn't provide a legal labels for the odds ratio: (%d,%d)" % (options.label1, options.label2))
      print(USAGE_STRING)
      sys.exit(2)

  if(options.classifier == "mostFrequent"):
    classifier = mostFrequent.MostFrequentClassifier(legalLabels)
  elif(options.classifier == "naiveBayes" or options.classifier == "nb"):
    classifier = naiveBayes.NaiveBayesClassifier(legalLabels)
    classifier.setSmoothing(options.smoothing)
    if (options.autotune):
        print("Using automatic tuning for naivebayes")
        classifier.automaticTuning = True
    else:
        print("Using smoothing parameter k=%f for naivebayes" %  options.smoothing)
  elif(options.classifier == "perceptron"):
    classifier = perceptron.PerceptronClassifier(legalLabels,options.iterations)
  elif(options.classifier == "kNN"):
    classifier = kNN.kNNClassifier(legalLabels, options.neighbors, options.data)
  elif(options.classifier == "mira"):
    classifier = mira.MiraClassifier(legalLabels, options.iterations)
    if (options.autotune):
        print("Using automatic tuning for MIRA")
        classifier.automaticTuning = True
    else:
        print("Using default C=0.001 for MIRA")
  elif(options.classifier == 'minicontest'):
    import minicontest
    classifier = minicontest.contestClassifier(legalLabels)
  else:
    print("Unknown classifier:", options.classifier)
    print(USAGE_STRING)
    
    sys.exit(2)

  print("====================================================================================================")

  args['classifier'] = classifier
  args['featureFunction'] = featureFunction
  args['printImage'] = printImage
  
  return args, options

USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python dataClassifier.py -c naiveBayes -d digits -t 1000 -f -o -1 3 -2 6 -k 2.5
                  - would run the naive Bayes classifier on 1000 training examples
                  using the enhancedFeatureExtractorDigits function to get the features
                  on the faces dataset, would use the smoothing parameter equals to 2.5, would
                  test the classifier on the test data and performs an odd ratio analysis
                  with label1=3 vs. label2=6
                 """

# Main harness code
import random
def runClassifier(args, options):

  featureFunction = args['featureFunction']
  classifier = args['classifier']
  printImage = args['printImage']
  ret_list = []
      
  # Load data  
  numTraining = options.training
  numTest = 0
  if options.data == "digits":
    numTest = options.testdigits
    rand_train = random.sample(range(TRAIN_SET_SIZE_DIGITS), numTraining)
    rand_val = random.sample(range(VALID_SET_SIZE_DIGITS), numTest)
    rand_test = random.sample(range(TEST_SET_SIZE_DIGITS), numTest)
    
  elif options.data == "faces":
    numTest = options.testfaces
    rand_train = random.sample(range(TRAIN_SET_SIZE_FACES), numTraining)
    rand_val = random.sample(range(VALID_SET_SIZE_FACES), numTest)
    rand_test = random.sample(range(TEST_SET_SIZE_FACES), numTest)

  # Load data file and label file. Read all the instances every time.
  if(options.data=="digits"):
    rawTrainingData = samples.loadDataFile("digitdata/trainingimages", TRAIN_SET_SIZE_DIGITS, DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", TRAIN_SET_SIZE_DIGITS)
    rawValidationData = samples.loadDataFile("digitdata/validationimages", VALID_SET_SIZE_DIGITS, DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("digitdata/validationlabels", VALID_SET_SIZE_DIGITS)
    rawTestData = samples.loadDataFile("digitdata/testimages", TEST_SET_SIZE_DIGITS, DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("digitdata/testlabels", TEST_SET_SIZE_DIGITS)
  else:
    rawTrainingData = samples.loadDataFile("facedata/facedatatrain", TRAIN_SET_SIZE_FACES, FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", TRAIN_SET_SIZE_FACES)
    rawValidationData = samples.loadDataFile("facedata/facedatavalidation", VALID_SET_SIZE_FACES, FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("facedata/facedatavalidationlabels", VALID_SET_SIZE_FACES)
    rawTestData = samples.loadDataFile("facedata/facedatatest", TEST_SET_SIZE_FACES, FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("facedata/facedatatestlabels", TEST_SET_SIZE_FACES)
    
  
  # Extract features
  print("Extracting features...")
  trainingData = map(featureFunction, rawTrainingData)
  validationData = map(featureFunction, rawValidationData)
  testData = map(featureFunction, rawTestData)
  
  # Randomly select numTraining examples for training and numTest examples for validation.
  # All the testing examples are used for the experiment.
  rand_trainingData = []
  rand_trainingLabels = []
  rand_validationData = []
  rand_validationLabels = []

  for i in rand_train:
    rand_trainingData.append(trainingData[i])
    rand_trainingLabels.append(trainingLabels[i])
  for i in rand_val:
    rand_validationData.append(validationData[i])
    rand_validationLabels.append(validationLabels[i])
  print("Training set used: ", len(rand_trainingData), "Testing set used: ", len(testData), "Number of Pixels: ", len(testData[0]))
  
  
  # Conduct training and testing
  print("====================================================================================================")  
  print("Training...")
  if options.classifier != "kNN":
    start_time = time.time()
    classifier.train(rand_trainingData, rand_trainingLabels, rand_validationData, rand_validationLabels)
    train_time = time.time() - start_time
    ret_list.append(train_time)
    print ("Training time: %s seconds ---" % train_time)
  else:
    start_time = time.time()
    classifier.train(rand_trainingData, rand_trainingLabels, rand_validationData, rand_validationLabels)
  
  print("====================================================================================================")
  print("Validating...")
  guesses = classifier.classify(rand_validationData)
  correct = [guesses[i] == rand_validationLabels[i] for i in range(len(rand_validationLabels))].count(True)
  print(str(correct), ("correct out of " + str(len(rand_validationLabels)) + " (%.1f%%).") % (100.0 * correct / len(rand_validationLabels)))

  print("====================================================================================================")
  print("Testing...")
  if options.classifier != "kNN":
    guesses = classifier.classify(testData)
  else:
    # start_time = time.time()
    guesses = classifier.classify(testData)
    test_time = time.time() - start_time
    ret_list.append(test_time)
    print ("Training time: %s seconds ---" % test_time)
  
  print("Predicted Label: {0}".format(guesses))
  print("True Label: {0}".format(testLabels))
  correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
  accuracy = float(correct)/float(len(testLabels))
  error = 1.0 - accuracy
  ret_list.append(accuracy)
  ret_list.append(error)
  print(str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * accuracy))
  # analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)
  
  # do odds ratio computation if specified at command line
  if((options.odds) & (options.classifier == "naiveBayes" or (options.classifier == "nb")) ):
    label1, label2 = options.label1, options.label2
    features_odds = classifier.findHighOddsFeatures(label1,label2)
    if(options.classifier == "naiveBayes" or options.classifier == "nb"):
      string3 = "=== Features with highest odd ratio of label %d over label %d ===" % (label1, label2)
    else:
      string3 = "=== Features for which weight(label %d)-weight(label %d) is biggest ===" % (label1, label2)    
      
    print(string3)
    printImage(features_odds)

  if((options.weights) & (options.classifier == "perceptron")):
    for l in classifier.legalLabels:
      features_weights = classifier.findHighWeightFeatures(l)
      print ("=== Features with high weight for label %d ==="%l)
      printImage(features_weights)
      
  print("====================================================================================================")
  print("====================================================================================================")
  print("====================================================================================================")
  return ret_list

      
if __name__ == '__main__':
  
  # Experiment  
  # perceptron_digits_list = []
  # perceptron_faces_list = []
  # perceptron_digits_list_5 = []
  # perceptron_faces_list_5 = []
  # nb_digits_list = []
  # nb_faces_list = []
  # kNN_digits_list = []
  # kNN_faces_list = []
  # kNN_digits_list_10 = []
  # kNN_faces_list_10 = []
  
  # for i in range(5):
    # print("Starting experiment {0} ...".format(i))
    # digit_train = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    # for numTrain in digit_train:
        # args, options = readCommand(["-c","perceptron", "-t", str(numTrain), "-s", "1000"])
        # perceptron_digits_list.append(runClassifier(args, options))
    # for numTrain in digit_train:
        # args, options = readCommand(["-c","perceptron", "-t", str(numTrain), "-s", "1000", "-i", "5"])
        # perceptron_digits_list_5.append(runClassifier(args, options))
    # for numTrain in digit_train:
        # args, options = readCommand(["-c", "naiveBayes", "-t", str(numTrain), "-s", "1000", "--autotune"])
        # nb_digits_list.append(runClassifier(args, options))
    # for numTrain in digit_train:
        # args, options = readCommand(["-c", "kNN", "-t", str(numTrain), "-s", "1000"])
        # kNN_digits_list.append(runClassifier(args, options))
    # for numTrain in digit_train:
        # args, options = readCommand(["-c", "kNN", "-t", str(numTrain), "-s", "1000", "-n", "10"])
        # kNN_digits_list_10.append(runClassifier(args, options))


    # faces_train = [45, 90, 135, 180, 225, 270, 315, 360, 405, 451]
    # for numTrain in faces_train:
        # args, options = readCommand(["-c","perceptron", "-d", "faces", "-t", str(numTrain), "-m", "150"])
        # perceptron_faces_list.append(runClassifier(args, options))
    # for numTrain in faces_train:
        # args, options = readCommand(["-c","perceptron", "-d", "faces", "-t", str(numTrain), "-m", "150", "-i", "5"])
        # perceptron_faces_list_5.append(runClassifier(args, options))
    # for numTrain in faces_train:
        # args, options = readCommand(["-c", "naiveBayes", "-d", "faces", "-t", str(numTrain), "-m", "150", "--autotune"])
        # nb_faces_list.append(runClassifier(args, options))
    # for numTrain in faces_train:
        # args, options = readCommand(["-c", "kNN", "-d", "faces", "-t", str(numTrain), "-m", "150"])
        # kNN_faces_list.append(runClassifier(args, options))
    # for numTrain in faces_train:
        # args, options = readCommand(["-c", "kNN", "-d", "faces", "-t", str(numTrain), "-m", "150", "-n", "10"])
        # kNN_faces_list_10.append(runClassifier(args, options))
  
  # print("perceptron_digits_list = {0}".format(perceptron_digits_list))
  # print("perceptron_digits_list_5 = {0}".format(perceptron_digits_list_5))
  # print("nb_digits_list = {0}".format(nb_digits_list))
  # print("kNN_digits_list = {0}".format(kNN_digits_list))
  # print("kNN_digits_list_10 = {0}".format(kNN_digits_list_10))
  # print("perceptron_faces_list = {0}".format(perceptron_faces_list))
  # print("perceptron_faces_list_5 = {0}".format(perceptron_faces_list_5))
  # print("nb_faces_list = {0}".format(nb_faces_list))
  # print("kNN_faces_list = {0}".format(kNN_faces_list))
  # print("kNN_faces_list_10 = {0}".format(kNN_faces_list_10))
  
  # Read input
  # args, options = readCommand(["-c","perceptron"]) 
  # args, options = readCommand(["-c", "naiveBayes", "--autotune" ])  
  args, options = readCommand( sys.argv[1:] )

  # Run classifier
  runClassifier(args, options)