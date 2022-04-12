
from __future__ import print_function

import re
import sys
import numpy as np
from operator import add
from pyspark import SparkContext
from pyspark.sql import SparkSession

sc = SparkContext.getOrCreate()
spark = SparkSession.builder.getOrCreate()

def buildArray(listOfIndices):
    returnVal = np.zeros(9000)   
    for index in listOfIndices:
        returnVal[index] = returnVal[index] + 1  
    mysum = np.sum(returnVal)  
    returnVal = np.divide(returnVal, mysum) 
    return returnVal

# Stops iterating through the list as soon as it finds the value
def getIndexOf(l, index, value):
    for pos,t in enumerate(l):
        if t[index] == value:
            return pos

def isNone(pos):
  if pos == None:
    return -1
  else:
    return pos

#Main
if __name__ == "__main__":

  # Use this code to read the data
  corpus = sc.textFile(sys.argv[1], 1)
  keyAndText = corpus.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))
  regex = re.compile('[^a-zA-Z]')
  keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', str(x[1])).lower().split()))

  #Task 1
  allWords = keyAndListOfWords.flatMap(lambda x: ( (j, 1) for j in x[1] ))

  # Now, count all of the words, giving us ("word1", 1433), ("word2", 3423423), etc.
  allCounts = allWords.reduceByKey(lambda a,b : a+b)

  # Get the top 10,000 words in a local array in a sorted format based on frequency
  # If you want to run it on your laptio, it may a longer time for top 20k words. 
  topWords = allCounts.top(9000, key=lambda x: x[1])

  topWordsK = sc.parallelize(range(9000))
  dictionary = topWordsK.map (lambda x : (topWords[x][0], x))

  print("Task 1: Using List:")
  print("Index for 'applicant' is",isNone(getIndexOf(topWords, 0, "applicant")))
  print("Index for 'and' is",isNone(getIndexOf(topWords, 0, "and")))
  print("Index for 'attack' is",isNone(getIndexOf(topWords, 0, "attack")))
  print("Index for 'protein' is",isNone(getIndexOf(topWords, 0, "protein")))
  print("Index for 'car' is",isNone(getIndexOf(topWords, 0, "car")))
  print("Index for 'in' is",isNone(getIndexOf(topWords, 0, "in")))

  print("\nTask 1: Using Dictionary:")
  print("Index for 'applicant' is",dictionary.filter(lambda x: x[0]=='applicant').take(1)[0][1])
  print("Index for 'and' is",dictionary.filter(lambda x: x[0]=='and').take(1)[0][1])
  print("Index for 'attack' is",dictionary.filter(lambda x: x[0]=='attack').take(1)[0][1])
  print("Index for 'protein' is",dictionary.filter(lambda x: x[0]=='protein').take(1)[0][1])
  print("Index for 'car' is",dictionary.filter(lambda x: x[0]=='car').take(1)[0][1])
  print("Index for 'in' is",dictionary.filter(lambda x: x[0]=='in').take(1)[0][1])

  #Task 2
  # Next, we get a RDD that has, for each (docID, ["word1", "word2", "word3", ...]),
  # ("word1", docID), ("word2", docId), ...

  allWordsWithDocID = keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))
  # Now join and link them, to get a set of ("word1", (dictionaryPos, docID)) pairs
  allDictionaryWords = dictionary.join(allWordsWithDocID)
  # Now, we drop the actual word itself to get a set of (docID, dictionaryPos) pairs
  justDocAndPos = allDictionaryWords.map (lambda x: (x[1][1], x[1][0]))
  # Now get a set of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
  allDictionaryWordsInEachDoc = justDocAndPos.groupByKey()
  allDocsAsNumpyArraysClassify = allDictionaryWordsInEachDoc.map(lambda x: (1 if 'AU' in x[0] else 0, buildArray(x[1])))
  traindata = allDocsAsNumpyArraysClassify.map(lambda x: (x[0],np.append(x[1],1)))
  traindata.cache()
  train_size = traindata.count()

  print("\nTask 2: Gradient Descent")

  # The optimised version of the code
  def LogisticRegression_optimized(traindata=traindata,
                        max_iteration = 5,
                        learningRate = 0.01,
                        regularization = 0.01,
                        mini_batch_size = 512,
                        tolerance = 10e-8,
                        beta = 0.9,
                        beta2 = 0.999,
                        optimizer = 'SGD',
                        train_size=train_size
                        ):

      # initialization
      prev_cost = 0
      L_cost = []
      prev_validation = 0

      parameter_size = len(traindata.take(1)[0][1])
      np.random.seed(0)
      parameter_vector = np.random.normal(0, 0.1, parameter_size)
      momentum = np.zeros(parameter_size)
      prev_mom = np.zeros(parameter_size)
      second_mom = np.array(parameter_size)
      gti = np.zeros(parameter_size)
      epsilon = 10e-8
      
      for i in range(max_iteration):

          bc_weights = parameter_vector

          min_batch = traindata.sample(False, mini_batch_size / train_size, 1 + i)
          
          res = min_batch.treeAggregate((np.zeros(parameter_size), 0, 0),\
                lambda x, y:(x[0]\
                            + (y[1]) * (-y[0] + (1/(np.exp(-np.dot(y[1], bc_weights))+1))),\
                            x[1] \
                            + y[0] * (-(np.dot(y[1], bc_weights))) \
                            + np.log(1 + np.exp(np.dot(y[1],bc_weights))),\
                            x[2] + 1),
                lambda x, y:(x[0] + y[0], x[1] + y[1], x[2] + y[2]))        

          gradients = res[0]
          sum_cost = res[1]
          num_samples = res[2]
          cost =  sum_cost/num_samples + regularization * (np.square(parameter_vector).sum())

          # calculate gradients
          gradient_derivative = (1.0 / num_samples) * gradients + 2 * regularization * parameter_vector
          
          if optimizer == 'SGD':
              parameter_vector = parameter_vector - learningRate * gradient_derivative          
              
          print("Iteration No.", i, " Cost=", cost)
          
          # Stop if the cost is not descreasing
          if abs(cost - prev_cost) < tolerance:
              print("cost - prev_cost: " + str(cost - prev_cost))
              break
          prev_cost = cost
          L_cost.append(cost)
          
      return parameter_vector, L_cost

  # Call of the optimzed function wiht the same parameters
  parameter_vector_sgd1, L_cost_sgd1 = LogisticRegression_optimized(traindata=traindata,
                        max_iteration = 50,
                        learningRate = 0.60,
                        regularization = 0.1,
                        mini_batch_size = 20000,
                        tolerance = .00005,
                        optimizer = 'SGD',
                        train_size = train_size
                        )

  indices = (-parameter_vector_sgd1).argsort()[:5]
  dictionaryswap = dictionary.map(lambda x: (x[1], x[0]))
  index_count = 1
  print("\nTask 2: Top 5 Words")
  for i in indices:
    print('No.', index_count, "Top Word:", dictionaryswap.filter(lambda x: x[0]==i).take(1)[0][1])
    index_count = index_count + 1

  #Task 3
  # Use this code to read the data
  corpustest = sc.textFile(sys.argv[2], 1)
  keyAndTexttest = corpustest.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))
  keyAndListOfWordstest = keyAndTexttest.map(lambda x : (str(x[0]), regex.sub(' ', str(x[1])).lower().split()))

  allWordsWithDocIDtest = keyAndListOfWordstest.flatMap(lambda x: ((j, x[0]) for j in x[1]))

  # Now join and link them, to get a set of ("word1", (dictionaryPos, docID)) pairs
  allDictionaryWordstest = dictionary.join(allWordsWithDocIDtest)

  justDocAndPostest = allDictionaryWordstest.map (lambda x: (x[1][1], x[1][0]))

  allDictionaryWordsInEachDoctest = justDocAndPostest.groupByKey()
  allDocsAsNumpyArraysClassifytest = allDictionaryWordsInEachDoctest.map(lambda x: (1 if 'AU' in x[0] else 0, buildArray(x[1])))

  testdata = allDocsAsNumpyArraysClassifytest.map(lambda x: (x[0],np.append(x[1],1)))

  result_test = testdata.map(lambda x: (x[0], np.dot(x[1], parameter_vector_sgd1))).map(lambda x: (x[0], 1 if x[1] >= 0.5 else 0))

  #TP, TN, FP, FN
  matrix = result_test.map(lambda x: (1 if x[0] + x[1] == 2 else 0, 1 if x[0] + x[1] == 0 else 0, 1 if x[0] < x[1] else 0, 1 if x[0] > x[1] else 0 )).reduce(lambda x, y: (x[0]+y[0], x[1]+y[1], x[2]+y[2], x[3]+y[3]))

  #TP/TP+FP
  if matrix[0]+matrix[2] == 0: predom = 1
  else: predom = matrix[0]+matrix[2]

  precision = matrix[0]/(predom)
  
  #TP/TP+FN
  recall = matrix[0]/(matrix[0]+matrix[3])

  #(2*(precision*recall)/(precision+recall))
  if precision+recall == 0: f1 = 0
  else: f1 = (2*(precision*recall))/(precision+recall)

  accuracy = (matrix[0]+matrix[1])/(matrix[0]+matrix[1]+matrix[2]+matrix[3])

  print("\nTask 3: Model Performance")
  print("TP:",matrix[0],", TN:",matrix[1],", FP:",matrix[2],", FN:", matrix[3])
  print("recall:", recall, ", precision:", precision, ", F1:", f1, ", accuracy:", accuracy)