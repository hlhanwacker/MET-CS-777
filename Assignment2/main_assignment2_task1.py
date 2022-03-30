import sys
import re
import numpy as np

from numpy import dot
from numpy.linalg import norm
from pyspark import SparkContext
from pyspark.sql import SparkSession
sc = SparkContext.getOrCreate()
#sc = SparkContext(appName="Hw2", conf=SparkConf().set('spark.driver.memory', '24g').set('spark.executor.memory', '12g'))
spark = SparkSession.builder.getOrCreate()

def buildArray(listOfIndices):
    
    returnVal = np.zeros(20000)
    
    for index in listOfIndices:
        returnVal[index] = returnVal[index] + 1
    
    mysum = np.sum(returnVal)
    
    returnVal = np.divide(returnVal, mysum)
    
    return returnVal


def build_zero_one_array (listOfIndices):
    
    returnVal = np.zeros (20000)
    
    for index in listOfIndices:
        if returnVal[index] == 0: returnVal[index] = 1
    
    return returnVal


def stringVector(x):
    returnVal = str(x[0])
    for j in x[1]:
        returnVal += ',' + str(j)
    return returnVal



def cousinSim (x,y):
	normA = np.linalg.norm(x)
	normB = np.linalg.norm(y)
	return np.dot(x,y)/(normA*normB)
 
#adding - function to help with ??? section
def zeroOneIt(x):
		x[x > 0] = 1
		return x



#Main
if __name__ == "__main__":

  # Set the file paths on your local machine
  # Change this line later on your python script when you want to run this on the CLOUD (GC or AWS)

  wikiPagesFile=sys.argv[1]
  wikiCategoryFile=sys.argv[2] 

  # Read two files into RDDs

  wikiCategoryLinks=sc.textFile(wikiCategoryFile)

  wikiCats=wikiCategoryLinks.map(lambda x: x.split(",")).map(lambda x: (x[0].replace('"', ''), x[1].replace('"', '') ))

  # Now the wikipages
  wikiPages = sc.textFile(wikiPagesFile)

  df = spark.read.csv(wikiPagesFile)
  # Assumption: Each document is stored in one line of the text file
  # We need this count later ... 
  numberOfDocs = wikiPages.count()

  print(numberOfDocs)
  # Each entry in validLines will be a line from the text file
  validLines = wikiPages.filter(lambda x : 'id' in x and 'url=' in x)

  # Now, we transform it into a set of (docID, text) pairs
  keyAndText = validLines.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6])) 
  
  # Now, we transform it into a set of (docID, text) pairs
  keyAndText = validLines.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))

  # Now, we split the text in each (docID, text) pair into a list of words
  # After this step, we have a data set with
  # (docID, ["word1", "word2", "word3", ...])
  # We use a regular expression here to make
  # sure that the program does not break down on some of the documents

  regex = re.compile('[^a-zA-Z]')

  # remove all non letter characters
  keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))
  # better solution here is to use NLTK tokenizer

  # Now get the top 20,000 words... first change (docID, ["word1", "word2", "word3", ...])
  # to ("word1", 1) ("word2", 1)...
  allWords = keyAndListOfWords.flatMap(lambda x: ( (j, 1) for j in x[1] ))

  # Now, count all of the words, giving us ("word1", 1433), ("word2", 3423423), etc.
  allCounts = allWords.reduceByKey(lambda a,b : a+b)

  # Get the top 20,000 words in a local array in a sorted format based on frequency
  # If you want to run it on your laptio, it may a longer time for top 20k words. 
  topWords = allCounts.top(20000, key=lambda x: x[1])

  # Top Words in Corpus:
  print("Top Words in Corpus:", allCounts.top(10, key=lambda x: x[1]))

  # We'll create a RDD that has a set of (word, dictNum) pairs
  # start by creating an RDD that has the number 0 through 20000
  # 20000 is the number of words that will be in our dictionary
  topWordsK = sc.parallelize(range(20000))

  # Now, we transform (0), (1), (2), ... to ("MostCommonWord", 1)
  # ("NextMostCommon", 2), ... 
  # the number will be the spot in the dictionary used to tell us
  # where the word is located
  dictionary = topWordsK.map (lambda x : (topWords[x][0], x))

  #Last 20 words in 20k positions
  print("Word Postions in our Feature Matrix. Last 20 words in 20k positions: ", dictionary.top(20, lambda x : x[1]))

  # Next, we get a RDD that has, for each (docID, ["word1", "word2", "word3", ...]),
  # ("word1", docID), ("word2", docId), ...

  allWordsWithDocID = keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))


  # Now join and link them, to get a set of ("word1", (dictionaryPos, docID)) pairs
  allDictionaryWords = dictionary.join(allWordsWithDocID)

  # Now, we drop the actual word itself to get a set of (docID, dictionaryPos) pairs
  justDocAndPos = allDictionaryWords.map (lambda x: (x[1][1], x[1][0]))


  # Now get a set of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
  allDictionaryWordsInEachDoc = justDocAndPos.groupByKey()


  # The following line this gets us a set of
  # (docID,  [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
  # and converts the dictionary positions to a bag-of-words numpy array...
  allDocsAsNumpyArrays = allDictionaryWordsInEachDoc.map(lambda x: (x[0], buildArray(x[1])))

  #savings output to argument - allDocsAsNumpyArrays.take(3)
  allDocsAsNumpyArraysresult = spark.sparkContext.parallelize(allDocsAsNumpyArrays.take(3))
  allDocsAsNumpyArraysresult.coalesce(1).saveAsTextFile(sys.argv[3])

  # Now, create a version of allDocsAsNumpyArrays where, in the array,
  # every entry is either zero or one.
  # A zero means that the word does not occur,
  # and a one means that it does.

  zeroOrOne = allDocsAsNumpyArrays.map(lambda x: (x[0], zeroOneIt(x[1])))

  # Now, add up all of those arrays into a single array, where the
  # i^th entry tells us how many
  # individual documents the i^th word in the dictionary appeared in
  dfArray = zeroOrOne.reduce(lambda x1, x2: ("", np.add(x1[1], x2[1])))[1]

  # Create an array of 20,000 entries, each entry with the value numberOfDocs (number of docs)
  multiplier = np.full(20000, numberOfDocs)

  # Get the version of dfArray where the i^th entry is the inverse-document frequency for the
  # i^th word in the corpus
  idfArray = np.log(np.divide(np.full(20000, numberOfDocs), dfArray)) 

  # Finally, convert all of the tf vectors in allDocsAsNumpyArrays to tf * idf vectors
  allDocsAsNumpyArraysTFidf = allDocsAsNumpyArrays.map(lambda x: (x[0], np.multiply(x[1], idfArray)))

  allDocsAsNumpyArraysTFidf.take(2)

  #savings output to argument - allDocsAsNumpyArraysTFidf.take(2)
  allDocsAsNumpyArraysTFidfresult = spark.sparkContext.parallelize(allDocsAsNumpyArraysTFidf.take(2))
  allDocsAsNumpyArraysTFidfresult.coalesce(1).saveAsTextFile(sys.argv[4])