from __future__ import print_function

import os
import sys
import requests
from operator import add

from pyspark import SparkConf,SparkContext
from pyspark.streaming import StreamingContext

from pyspark.sql import SparkSession
from pyspark.sql import SQLContext

from pyspark.sql.types import *
from pyspark.sql import functions as func
from pyspark.sql.functions import *

spark = SparkSession.builder.master("local[*]").getOrCreate()
sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)

#Exception Handling and removing wrong datalines
def isfloat(value):
    try:
        float(value)
        return True
 
    except:
         return False

#Function - Cleaning
#For example, remove lines if they don’t have 16 values and 
# checking if the trip distance and fare amount is a float number
# checking if the trip duration is more than a minute, trip distance is more than 0.1 miles, 
# fare amount and total amount are more than 0.1 dollars
def correctRows(p):
    if(len(p)==17):
        if(isfloat(p[5]) and isfloat(p[11])):
            if(float(p[4])> 60 and float(p[5])>0.10 and float(p[11])> 0.10 and float(p[16])> 0.10):
                return p

#Main
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: main_task1 <file> <output> ", file=sys.stderr)
        exit(-1)
  
    # Create a DataFrame
    data = spark.read.csv(sys.argv[1])

    testRDD = data.rdd.map(tuple)

    # calling correctRows and isfloat functions to cleaning up data
    taxilinesCorrected = testRDD.filter(correctRows)

    #mapping data
    taxi = taxilinesCorrected.map(lambda p: (p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15], p[16]))

    driveravg = taxi.map(lambda p: (p[1],(((float(p[16]))/(float(p[4]))*60),1))).reduceByKey(lambda x,y: (x[0]+y[0], x[1]+y[1])).map(lambda x: (x[0], x[1][0]/x[1][1]))

    #getting top 10 drivers by average money per minute
    driveravgtop = driveravg.top(10, lambda x:x[1])

    #converting list to rdd
    driveravgtoprdd = spark.sparkContext.parallelize(driveravgtop)

    #savings output to argument
    driveravgtoprdd.coalesce(1).saveAsTextFile(sys.argv[2])    