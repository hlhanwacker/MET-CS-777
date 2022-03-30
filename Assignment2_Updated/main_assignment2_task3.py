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

  #converting RDD to spark DF
  wikiCatsdf = wikiCats.toDF()

  wikiCatsdf2 = wikiCatsdf.withColumnRenamed("_1","DocID").withColumnRenamed("_2","Category")

  #viewing data
  wikiCatsdf2.head(10)

  #viewing column names
  list(wikiCatsdf.columns)
  list(wikiCatsdf2.columns)

  #grouping by category and grouping by doc
  numcategoriesperdoc = wikiCatsdf2.groupby(['DocID']).count()
  numdocspercategory = wikiCatsdf2.groupby(['Category']).count()
  
  #summary stats
  results = numcategoriesperdoc.describe(['count']).collect()
  resultsnumcatperdoc =sc.parallelize(results).coalesce(1)
  resultsnumcatperdoc.saveAsTextFile(sys.argv[3])

  #median
  import pyspark.sql.functions as func
  median = numcategoriesperdoc.agg(func.percentile_approx('count', 0.5)).collect()
  resultsmedian =sc.parallelize(median).coalesce(1)
  resultsmedian.saveAsTextFile(sys.argv[4])

  #Top 10 Category
  categorybycount = numdocspercategory.orderBy('count', ascending=[False]).head(10)
  categorybycountresults = sc.parallelize(categorybycount).coalesce(1)
  categorybycountresults.saveAsTextFile(sys.argv[5])