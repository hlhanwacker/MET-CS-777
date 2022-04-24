from __future__ import print_function

import os
import re
import sys
import time
import requests
from operator import add
import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import functions as func
from pyspark.sql.functions import *  
from pyspark.ml.feature import StringIndexer
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
sc = SparkContext.getOrCreate()
spark = SparkSession.builder.getOrCreate()
sqlContext = SQLContext(sc)

if __name__ == "__main__":
  # Create a DataFrame
  df = spark.read.option("header", True).csv(sys.argv[1])
  #column names in dataframe
  df.columns
  #Show raw data
  print("\nImported Data:")
  df.show()
  #removing columns with null values
  dfCorrected = df.na.drop()
  #Show corrected data
  print("\nData after removing nulls:")
  dfCorrected.show()
  #summary statistics on attributes (ignore Date)
  print("\nSummary Statistics:")
  dfCorrected.summary().show()
  #summary statistics on attributes for 2021 (ignore Date)
  print("\nSummary Statistics 2021:")
  dfCorrected.filter(col("Date").contains("2021")).summary().show()
  #summary statistics on attributes for 2008 (ignore Date)
  print("\nSummary Statistics 2008:")
  dfCorrected.filter(col("Date").contains("2008")).summary().show()
  #show df schema - note all attributes showing as string
  print("\nSchema:")
  dfCorrected.printSchema()
  #cast attributes being used for analysis to float
  df2 = dfCorrected.withColumn("DOW_Adj Close", dfCorrected["DOW_Adj Close"].cast(FloatType())).withColumn("HPI_USA_SA", dfCorrected["HPI_USA_SA"].cast(FloatType())).withColumn("FRED_30_Mortgage_US", dfCorrected["FRED_30_Mortgage_US"].cast(FloatType())).withColumn("FRED_Median_Sale_Price", dfCorrected["FRED_Median_Sale_Price"].cast(FloatType())).withColumn("US_Gas_AllFormulations_Price_per_Gallon", dfCorrected["US_Gas_AllFormulations_Price_per_Gallon"].cast(FloatType()))
  #print df schema after correction
  print("\nSchema after casting:")
  df2.printSchema()

  #convert string to numeric feature and create a new dataframe
  #new dataframe contains a new feature 'date_cat' and can be used further
  #feature date_cat is now vectorized and can be used to fed to model
  indexer=StringIndexer(inputCol='Date',outputCol='Date_cat')
  indexed=indexer.fit(df2).transform(df2)
  print("Indexing Date:")
  for item in indexed.head(5):
      print(item)

  #**MODEL WITH ALL ATTRIBUTES**#

  #creating vectors from features
  #MLlib takes input if vector form
  assembler = VectorAssembler(inputCols=['DOW_Adj Close', 'HPI_USA_SA', 'FRED_30_Mortgage_US', 'US_Gas_AllFormulations_Price_per_Gallon', 'Date_cat'],outputCol='features')
  output=assembler.transform(indexed)
  print("\nVectorized Data:")
  output.select('features','FRED_Median_Sale_Price').show(5)

  #final data consist of features and label which is median sale price.
  final_data=output.select('features','FRED_Median_Sale_Price')
  #splitting data into train and test
  train_data,test_data=final_data.randomSplit([0.7,0.3])
  print("\nTrain Data:")
  train_data.describe().show()
  print("\nTest Data:")
  test_data.describe().show()

  #creating an object of class LinearRegression
  #object takes features and label as input arguments
  data_lr=LinearRegression(featuresCol='features',labelCol='FRED_Median_Sale_Price',solver='normal', maxIter=100, loss='squaredError')
  #pass train_data to train model
  trained_model=data_lr.fit(train_data)
  #evaluating model trained for Rsquared error
  results=trained_model.evaluate(train_data)
  
  print('\nModel with All Attributes:')
  print('Rsquared Error:',results.r2)
  print('Mean Squared Error :',results.meanSquaredError)
  print('Root Mean Squared Error :',results.rootMeanSquaredError)
  print('Mean Absolute Error :',results.meanAbsoluteError)

  #**MODEL WITHOUT Date ATTRIBUTE**#

  #creating vectors from features
  #MLlib takes input if vector form
  assembler = VectorAssembler(inputCols=['DOW_Adj Close', 'HPI_USA_SA', 'FRED_30_Mortgage_US', 'US_Gas_AllFormulations_Price_per_Gallon'],outputCol='features')
  output=assembler.transform(indexed)

  #final data consist of features and label which is median sale price.
  final_data=output.select('features','FRED_Median_Sale_Price')
  #splitting data into train and test
  train_data,test_data=final_data.randomSplit([0.7,0.3])

  #creating an object of class LinearRegression
  #object takes features and label as input arguments
  data_lr=LinearRegression(featuresCol='features',labelCol='FRED_Median_Sale_Price',solver='normal', maxIter=100, loss='squaredError')
  #pass train_data to train model
  trained_model=data_lr.fit(train_data)
  #evaluating model trained for Rsquared error
  results=trained_model.evaluate(train_data)
  
  print('\nModel without Date:')
  print('Rsquared Error:',results.r2)
  print('Mean Squared Error :',results.meanSquaredError)
  print('Root Mean Squared Error :',results.rootMeanSquaredError)
  print('Mean Absolute Error :',results.meanAbsoluteError)

  #**MODEL WITHOUT Date ATTRIBUTE and Gas Price**#

  #creating vectors from features
  #MLlib takes input if vector form
  assembler = VectorAssembler(inputCols=['DOW_Adj Close', 'HPI_USA_SA', 'FRED_30_Mortgage_US'],outputCol='features')
  output=assembler.transform(indexed)

  #final data consist of features and label which is median sale price.
  final_data=output.select('features','FRED_Median_Sale_Price')
  #splitting data into train and test
  train_data,test_data=final_data.randomSplit([0.7,0.3])

  #creating an object of class LinearRegression
  #object takes features and label as input arguments
  data_lr=LinearRegression(featuresCol='features',labelCol='FRED_Median_Sale_Price',solver='normal', maxIter=100, loss='squaredError')
  #pass train_data to train model
  trained_model=data_lr.fit(train_data)
  #evaluating model trained for Rsquared error
  results=trained_model.evaluate(train_data)
  
  print('\nModel without Date and Gas Prices:')
  print('Rsquared Error:',results.r2)
  print('Mean Squared Error :',results.meanSquaredError)
  print('Root Mean Squared Error :',results.rootMeanSquaredError)
  print('Mean Absolute Error :',results.meanAbsoluteError)

  #**MODEL WITHOUT Date ATTRIBUTE, Interest Rates and Gas Price**#

  #creating vectors from features
  #MLlib takes input if vector form
  assembler = VectorAssembler(inputCols=['DOW_Adj Close', 'HPI_USA_SA'],outputCol='features')
  output=assembler.transform(indexed)

  #final data consist of features and label which is median sale price.
  final_data=output.select('features','FRED_Median_Sale_Price')
  #splitting data into train and test
  train_data,test_data=final_data.randomSplit([0.7,0.3])

  #creating an object of class LinearRegression
  #object takes features and label as input arguments
  data_lr=LinearRegression(featuresCol='features',labelCol='FRED_Median_Sale_Price',solver='normal', maxIter=100, loss='squaredError')
  #pass train_data to train model
  trained_model=data_lr.fit(train_data)
  #evaluating model trained for Rsquared error
  results=trained_model.evaluate(train_data)
  
  print('\nModel only DOW and HPI:')
  print('Rsquared Error:',results.r2)
  print('Mean Squared Error :',results.meanSquaredError)
  print('Root Mean Squared Error :',results.rootMeanSquaredError)
  print('Mean Absolute Error :',results.meanAbsoluteError)

  #**MODEL ONLY HPI**#

  #creating vectors from features
  #MLlib takes input if vector form
  assembler = VectorAssembler(inputCols=['HPI_USA_SA'],outputCol='features')
  output=assembler.transform(indexed)

  #final data consist of features and label which is median sale price.
  final_data=output.select('features','FRED_Median_Sale_Price')
  #splitting data into train and test
  train_data,test_data=final_data.randomSplit([0.7,0.3])

  #creating an object of class LinearRegression
  #object takes features and label as input arguments
  data_lr=LinearRegression(featuresCol='features',labelCol='FRED_Median_Sale_Price',solver='normal', maxIter=100, loss='squaredError')
  #pass train_data to train model
  trained_model=data_lr.fit(train_data)
  #evaluating model trained for Rsquared error
  results=trained_model.evaluate(train_data)
  
  print('\nModel only HPI:')
  print('Rsquared Error:',results.r2)
  print('Mean Squared Error :',results.meanSquaredError)
  print('Root Mean Squared Error :',results.rootMeanSquaredError)
  print('Mean Absolute Error :',results.meanAbsoluteError)

  #**MODEL ONLY INTEREST RATES**#

  #creating vectors from features
  #MLlib takes input if vector form
  assembler = VectorAssembler(inputCols=['FRED_30_Mortgage_US'],outputCol='features')
  output=assembler.transform(indexed)

  #final data consist of features and label which is median sale price.
  final_data=output.select('features','FRED_Median_Sale_Price')
  #splitting data into train and test
  train_data,test_data=final_data.randomSplit([0.7,0.3])

  #creating an object of class LinearRegression
  #object takes features and label as input arguments
  data_lr=LinearRegression(featuresCol='features',labelCol='FRED_Median_Sale_Price',solver='normal', maxIter=100, loss='squaredError')
  #pass train_data to train model
  trained_model=data_lr.fit(train_data)
  #evaluating model trained for Rsquared error
  results=trained_model.evaluate(train_data)
  
  print('\nModel only Interest Rates:')
  print('Rsquared Error:',results.r2)
  print('Mean Squared Error :',results.meanSquaredError)
  print('Root Mean Squared Error :',results.rootMeanSquaredError)
  print('Mean Absolute Error :',results.meanAbsoluteError)

  #**MODEL ONLY DOW**#

  #creating vectors from features
  #MLlib takes input if vector form
  assembler = VectorAssembler(inputCols=['DOW_Adj Close'],outputCol='features')
  output=assembler.transform(indexed)

  #final data consist of features and label which is median sale price.
  final_data=output.select('features','FRED_Median_Sale_Price')
  #splitting data into train and test
  train_data,test_data=final_data.randomSplit([0.7,0.3])

  #creating an object of class LinearRegression
  #object takes features and label as input arguments
  data_lr=LinearRegression(featuresCol='features',labelCol='FRED_Median_Sale_Price',solver='normal', maxIter=100, loss='squaredError')
  #pass train_data to train model
  trained_model=data_lr.fit(train_data)
  #evaluating model trained for Rsquared error
  results=trained_model.evaluate(train_data)
  
  print('\nModel only DOW:')
  print('Rsquared Error:',results.r2)
  print('Mean Squared Error :',results.meanSquaredError)
  print('Root Mean Squared Error :',results.rootMeanSquaredError)
  print('Mean Absolute Error :',results.meanAbsoluteError)
