
# coding: utf-8

# In[263]:


import pandas as pd
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SQLContext
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType
from pyspark.ml.feature import Word2Vec
from pyspark.sql.functions import udf
from pyspark.sql.functions import col, when
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SQLContext
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, DoubleType
from pyspark.ml.feature import Word2Vec
from pyspark.ml.feature import HashingTF, Tokenizer
import pandas as pd
from pyspark.sql import SQLContext
from pyspark.sql import Row, functions
from pyspark.sql.functions import udf, col
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SQLContext
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType
from pyspark.ml.feature import Word2Vec
from pyspark.sql import Row, functions
from pyspark.sql.functions import udf, col
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import LinearRegression
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import StopWordsRemover
import math
import numpy as np
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import RandomForestRegressor



# In[264]:


# mergefunction to use only brand, color and material in product_attributes
def mergeFunction(attr):
    names = attributes["name"]
    values = attributes["value"]
    result = []
    for name, value in zip(names, values):
        if "Brand".lower() in name.lower():
            result.append(value)
    return " ".join(result)


# In[265]:


# Function to find cosine similarity between vectors
def cosineSimilarity(v1,v2):
    sumOfXs = 0
    sumOfYs = 0
    sumOfXYs = 0
    for i in range(len(v1)):
        sumOfXs += v1[i] * v1[i]
        sumOfYs +=  v2[i] * v2[i]
        sumOfXYs += v2[i] * v1[i]
        
    return float(sumOfXYs / math.sqrt(sumOfXs * sumOfYs))

idfUDF=udf(cosineSimilarity, DoubleType())


# In[266]:


# Function to find euclidean distance between vectors
def euclideanDistance(v1,v2):
    dist = 0
    if len(v1) < len(v2):
        for i in range(len(v2)):
            if i < len(v1):
                dist += (v2[i] - v1[i]) * (v2[i] - v1[i])
            else:
                dist += v2[i] * v2[i]
    else:
        for i in range(len(v1)):
            if i < len(v2):
                dist += (v2[i] - v1[i]) * (v2[i] - v1[i])
            else:
                dist += v1[i] * v1[i]
    return float(math.sqrt(dist))

edUDF=udf(euclideanDistance, DoubleType())
# v1 = [10, 10, 10]
# v2 = [5, 5]
# print (euclideanDistance(v1, v2))


# In[267]:


# Function to find how many words match between vectors
def numberOfWordsMatched(v1,v2):
    l1 = len(v1)
    l2 = len(v2)
    match = 0
    for i in range(l1):
        v1[i] = v1[i].lower()
        for j in range(l2):
            v2[j] = v2[j].lower()
            if v1[i] == v2[j]:
                match += 2
            elif v1[i] in v2[j]:
                match += 1
            elif v2[j] in v1[i]:
                match += 1
            else:
                match += 0
    return match
matchUDF=udf(numberOfWordsMatched, IntegerType())

# v1 = ["hi", "my", "name"]
# v2 = ["hi", "abc", "names"]
# print (numberOfWordsMatched(v1,v2))


# In[268]:


# Import all files
test_data = pd.read_csv("/home/jiawenz1_c4gcp/test.csv", encoding = 'ISO-8859-1')
train_data = pd.read_csv('/home/jiawenz1_c4gcp/train.csv', encoding = 'ISO-8859-1')
product_description = pd.read_csv('/home/jiawenz1_c4gcp/product_descriptions.csv', encoding = 'ISO-8859-1')
attributes = pd.read_csv('/home/jiawenz1_c4gcp/attributes.csv', encoding = 'ISO-8859-1')


# In[269]:


# merge train dataframe and description dataframe
train_and_description = pd.merge(train_data, product_description, how="left", on="product_uid")

attributes.dropna(how="all", inplace=True)
attributes["product_uid"] = attributes["product_uid"].astype(int)

attributes["value"] = attributes["value"].astype(str)

fields = [StructField("product_uid", StringType(), True), StructField("name", StringType(), True), StructField("value", StringType(), True)]
mergeSchema = StructType(fields)
product_attributes = sqlContext.createDataFrame(attributes, mergeSchema)
product_attributes = product_attributes.select("product_uid", "name", col("value").alias("product_attributes")).filter(product_attributes["name"] == "MFG Brand Name").drop("name")
product_attributes.show(5)


# In[254]:



# merge train_description dataframe and attribute dataframe
train_description_attributes = train_and_description

train_description_attributes['search_term_length'] = train_description_attributes['search_term'].map(lambda x:len(x.split())).astype(np.int64)

fields = [StructField("id", StringType(), True), StructField("product_uid", StringType(), True), StructField("product_title", StringType(), True), StructField("search_term", StringType(), True)
         , StructField("relevance", StringType(), True), StructField("product_description", StringType(), True), StructField("search_term_length", IntegerType(), True)]
mergeSchema = StructType(fields)

# convert pandas dataframe to spark dataframe
training = sqlContext.createDataFrame(train_description_attributes, mergeSchema)
training = training.join(product_attributes, training.product_uid == product_attributes.product_uid, "left").drop(product_attributes.product_uid)
training = training.withColumn("product_attributes", when(col("product_attributes").isNull(), "empty").otherwise(col("product_attributes")))
training.show(5)


# In[255]:


# # Processing Dataframe
tokenizer = Tokenizer(inputCol="search_term", outputCol="search_term_")
training = tokenizer.transform(training)
tokenizer = Tokenizer(inputCol="product_title", outputCol="product_title_")
training = tokenizer.transform(training)
tokenizer = Tokenizer(inputCol="product_description", outputCol="product_description__")
training = tokenizer.transform(training)
remover = StopWordsRemover(inputCol="product_description__", outputCol="product_description_")
training = remover.transform(training)
tokenizer = Tokenizer(inputCol="product_attributes", outputCol="product_attributes_")
training = tokenizer.transform(training)
to_double = training[4].cast(DoubleType())
training = training.withColumn('relevance', to_double)
training = training.drop('search_term').drop('product_title').drop('product_description').drop('product_attributes').drop('product_description__')

training.show(5)


# In[256]:


# Use word2Vec to do feature extraction
word2Vec = Word2Vec(vectorSize=10, minCount=0, inputCol="search_term_", outputCol="search_term_array_features")
model = word2Vec.fit(training)
training = model.transform(training)
word2Vec = Word2Vec(vectorSize=10, minCount=0, inputCol="product_title_", outputCol="product_title_array_features")
model = word2Vec.fit(training)
training = model.transform(training)
word2Vec = Word2Vec(vectorSize=10, minCount=0, inputCol="product_attributes_", outputCol="product_attributes_array_features")
model = word2Vec.fit(training)
training = model.transform(training)

training.show(5)


# In[257]:



# call the idfUDF function to calculate the Cosine Similarity between each two vectors
training = training.withColumn("searchAndTitle", idfUDF("search_term_array_features","product_title_array_features"))
training = training.withColumn("searchAndAttributes", idfUDF("search_term_array_features","product_attributes_array_features"))

# call the edUDF function to calculate the Eucledian Distance between each two vectors
training = training.withColumn("searchAndTitleED", edUDF("search_term_array_features","product_title_array_features"))
training = training.withColumn("searchAndAttributesED", edUDF("search_term_array_features","product_attributes_array_features"))

# call the matchUDF function to calculate the Eucledian Distance between each two vectors
training = training.withColumn("searchAndTitleMD", matchUDF("search_term_","product_title_"))
training = training.withColumn("searchAndAttributesMD", matchUDF("search_term_","product_attributes_"))
training = training.withColumn("searchAndDescriptionMD", matchUDF("search_term_","product_description_"))

# combine three features into a dictionary
features=["searchAndTitleMD", "searchAndAttributesMD", "searchAndDescriptionMD", "search_term_length", "searchAndTitleED", "searchAndAttributesED", "searchAndTitle", "searchAndAttributes"]
assembler_features = VectorAssembler(inputCols=features, outputCol='features')
training = assembler_features.transform(training)
training.select("features").show(10)


# In[258]:


# Processing the testing data, tokeninzing columns
test_and_description = pd.merge(test_data, product_description, how="left", on="product_uid")

test_and_description['search_term_length'] = test_and_description['search_term'].map(lambda x:len(x.split())).astype(np.int64)

fields = [StructField("id", StringType(), True), StructField("product_uid", StringType(), True), StructField("product_title", StringType(), True), StructField("search_term", StringType(), True)
         , StructField("product_description", StringType(), True), StructField("search_term_length", IntegerType(), True)]
mergeSchema = StructType(fields)
testing = sqlContext.createDataFrame(test_and_description, mergeSchema)

testing = testing.join(product_attributes, testing.product_uid == product_attributes.product_uid, "left").drop(product_attributes.product_uid)
testing = testing.withColumn("product_attributes", when(col("product_attributes").isNull(), "empty").otherwise(col("product_attributes")))


# # Processing Dataframe
tokenizer = Tokenizer(inputCol="search_term", outputCol="search_term_")
testing = tokenizer.transform(testing)
tokenizer = Tokenizer(inputCol="product_title", outputCol="product_title_")
testing = tokenizer.transform(testing)
tokenizer = Tokenizer(inputCol="product_description", outputCol="product_description__")
testing = tokenizer.transform(testing)
remover = StopWordsRemover(inputCol="product_description__", outputCol="product_description_")
testing = remover.transform(testing)
tokenizer = Tokenizer(inputCol="product_attributes", outputCol="product_attributes_")
testing = tokenizer.transform(testing)
testing = testing.drop('search_term').drop('product_title').drop('product_description').drop('product_attributes').drop('product_attributes__')
testing.show(5)


# In[259]:


# Converting all tokens to vectors
word2Vec = Word2Vec(vectorSize=10, minCount=0, inputCol="search_term_", outputCol="search_term_array_features")
model = word2Vec.fit(testing)
testing = model.transform(testing)
word2Vec = Word2Vec(vectorSize=10, minCount=0, inputCol="product_title_", outputCol="product_title_array_features")
model = word2Vec.fit(testing)
testing = model.transform(testing)
word2Vec = Word2Vec(vectorSize=10, minCount=0, inputCol="product_attributes_", outputCol="product_attributes_array_features")
model = word2Vec.fit(testing)
testing = model.transform(testing)
testing.show(5)


# In[260]:


# Finding Cosine Similarity between vectors
testing = testing.withColumn("searchAndTitle", idfUDF("search_term_array_features","product_title_array_features"))
testing = testing.withColumn("searchAndAttributes", idfUDF("search_term_array_features","product_attributes_array_features"))

# call the edUDF function to calculate the Eucledian Distance between each two vectors
testing = testing.withColumn("searchAndTitleED", edUDF("search_term_array_features","product_title_array_features"))
testing = testing.withColumn("searchAndAttributesED", edUDF("search_term_array_features","product_attributes_array_features"))

# call the matchUDF function to calculate the Eucledian Distance between each two vectors
testing = testing.withColumn("searchAndTitleMD", matchUDF("search_term_","product_title_"))
testing = testing.withColumn("searchAndAttributesMD", matchUDF("search_term_","product_attributes_"))
testing = testing.withColumn("searchAndDescriptionMD", matchUDF("search_term_","product_description_"))

# combine seven features into a dictionary
#features=[]
features=["searchAndTitleMD", "searchAndAttributesMD", "searchAndDescriptionMD", "search_term_length", "searchAndTitleED", "searchAndAttributesED", "searchAndTitle", "searchAndAttributes"]
assembler_features = VectorAssembler(inputCols=features, outputCol='features')
testing = assembler_features.transform(testing)
testing.show(5)
testing.select("features").show(10)


# In[261]:


# Using thr Random Forest Regressor on the data
rf = RandomForestRegressor(featuresCol="features",labelCol='relevance', numTrees=200, maxDepth=10)
pipeline = Pipeline(stages=[rf])

# train the model
model = pipeline.fit(training)

# testing
prediction = model.transform(testing)
prediction.show(5)


# In[262]:


print (prediction)
# Writing prediction to a csv file on instance
print (training.count())
submission=prediction.select("id","prediction")
submission.show(100)
print (submission.count())
submission = submission.toPandas()
# output the result into a csv file
submission.to_csv('/home/jiawenz1_c4gcp/answer.csv', index=False)


# In[ ]:




