
# coding: utf-8

# In[24]:


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
import math
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import RandomForestRegressor



# In[25]:


# mergefunction to use only brand, color and material in product_attributes
def mergeFunction(attr):
    names = attributes["name"]
    values = attributes["value"]
    result = []
    for name, value in zip(names, values):
        if "Brand".lower() or "Color".lower() or "Material".lower() in name.lower():
            result.append(value)
    return " ".join(result)


# In[26]:


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


# In[37]:


# Import all files
test_data = pd.read_csv("/home/jiawenz1_c4gcp/test.csv", encoding = 'ISO-8859-1').head(100)
train_data = pd.read_csv('/home/jiawenz1_c4gcp/train.csv', encoding = 'ISO-8859-1').head(100)
product_description = pd.read_csv('/home/jiawenz1_c4gcp/product_descriptions.csv', encoding = 'ISO-8859-1').head(100)
attributes = pd.read_csv('/home/jiawenz1_c4gcp/attributes.csv', encoding = 'ISO-8859-1').head(100)


# In[38]:



# merge train dataframe and description dataframe
train_and_description = pd.merge(train_data, product_description, how="left", on="product_uid")
attributes.dropna(how="all", inplace=True)
attributes["product_uid"] = attributes["product_uid"].astype(int)
attributes["value"] = attributes["value"].astype(str)
product_attributes = attributes.groupby("product_uid").apply(mergeFunction)
product_attributes = product_attributes.reset_index(name="product_attributes")


# In[39]:



# merge train_description dataframe and attribute dataframe
train_description_attributes = pd.merge(train_and_description, product_attributes, how="left", on="product_uid")
fields = [StructField("id", StringType(), True), StructField("product_uid", StringType(), True), StructField("product_title", StringType(), True), StructField("search_term", StringType(), True)
         , StructField("relevance", StringType(), True), StructField("product_description", StringType(), True), StructField("product_attributes", StringType(), True)]
mergeSchema = StructType(fields)
# convert pandas dataframe to spark dataframe
training = sqlContext.createDataFrame(train_description_attributes, mergeSchema)
training.show(5)


# In[40]:



# # Processing Dataframe
split_col = functions.split(training[3], " ")
training = training.withColumn('search_term', split_col)
split_col = functions.split(training[2], " ")
training = training.withColumn('product_title', split_col)
split_col = functions.split(training[5], " ")
training = training.withColumn('product_description', split_col)
split_col = functions.split(training[6], " ")
training = training.withColumn('product_attributes', split_col)
to_double = training[4].cast(DoubleType())
training = training.withColumn('relevance', to_double)
training.show(5)


# In[41]:


# Use word2Vec to do feature extraction
word2Vec = Word2Vec(vectorSize=10, minCount=0, inputCol="search_term", outputCol="search_term_array_features")
model = word2Vec.fit(training)
training = model.transform(training)
word2Vec = Word2Vec(vectorSize=10, minCount=0, inputCol="product_title", outputCol="product_title_array_features")
model = word2Vec.fit(training)
training = model.transform(training)
word2Vec = Word2Vec(vectorSize=10, minCount=0, inputCol="product_description", outputCol="product_description_array_features")
model = word2Vec.fit(training)
training = model.transform(training)
word2Vec = Word2Vec(vectorSize=10, minCount=0, inputCol="product_attributes", outputCol="product_attributes_array_features")
model = word2Vec.fit(training)
training = model.transform(training)
training.show(5)


# In[42]:



# call the idfUDF function to calculate the Cosine Similarity between each two vectors
training = training.withColumn("searchAndTitle", idfUDF("search_term_array_features","product_title_array_features"))
training = training.withColumn("searchAndDescription", idfUDF("search_term_array_features","product_description_array_features"))
training = training.withColumn("searchAndAttributes", idfUDF("search_term_array_features","product_attributes_array_features"))


# combine three features into a dictionary
features=["searchAndTitle", "searchAndDescription", "searchAndAttributes"]
assembler_features = VectorAssembler(inputCols=features, outputCol='features')
training = assembler_features.transform(training)
training.show(5)
training.select("features").show(10)


# In[43]:


# Processing the testing data, tokeninzing columns
test_and_description = pd.merge(test_data, product_description, how="left", on="product_uid")
test_description_attributes = pd.merge(test_and_description, product_attributes, how="left", on="product_uid")
fields = [StructField("id", StringType(), True), StructField("product_uid", StringType(), True), StructField("product_title", StringType(), True), StructField("search_term", StringType(), True)
         , StructField("product_description", StringType(), True), StructField("product_attributes", StringType(), True)]
mergeSchema = StructType(fields)
testing = sqlContext.createDataFrame(test_description_attributes, mergeSchema)
split_col = functions.split(testing[3], " ")
testing = testing.withColumn('search_term', split_col)
split_col = functions.split(testing[2], " ")
testing = testing.withColumn('product_title', split_col)
split_col = functions.split(testing[4], " ")
testing = testing.withColumn('product_description', split_col)
split_col = functions.split(testing[5], " ")
testing = testing.withColumn('product_attributes', split_col)
testing.show(5)


# In[44]:


# Converting all tokens to vectors
word2Vec = Word2Vec(vectorSize=10, minCount=0, inputCol="search_term", outputCol="search_term_array_features")
model = word2Vec.fit(testing)
testing = model.transform(testing)
word2Vec = Word2Vec(vectorSize=10, minCount=0, inputCol="product_title", outputCol="product_title_array_features")
model = word2Vec.fit(testing)
testing = model.transform(testing)
word2Vec = Word2Vec(vectorSize=10, minCount=0, inputCol="product_description", outputCol="product_description_array_features")
model = word2Vec.fit(testing)
testing = model.transform(testing)
word2Vec = Word2Vec(vectorSize=10, minCount=0, inputCol="product_attributes", outputCol="product_attributes_array_features")
model = word2Vec.fit(testing)
testing = model.transform(testing)
testing.show(5)


# In[45]:


# Finding Cosine Similarity between vectors
testing = testing.withColumn("searchAndTitle", idfUDF("search_term_array_features","product_title_array_features"))
testing = testing.withColumn("searchAndDescription", idfUDF("search_term_array_features","product_description_array_features"))
testing = testing.withColumn("searchAndAttributes", idfUDF("search_term_array_features","product_attributes_array_features"))
features=["searchAndTitle", "searchAndDescription", "searchAndAttributes"]
assembler_features = VectorAssembler(inputCols=features, outputCol='features')
testing = assembler_features.transform(testing)
testing.show(5)
testing.select("features").show(10)


# In[46]:


# Using thr Random Forest Regressor on the data
rf = RandomForestRegressor(featuresCol="features",labelCol='relevance', numTrees=20, maxDepth=5)
pipeline = Pipeline(stages=[rf])

# train the model
model = pipeline.fit(training)

# testing
prediction = model.transform(testing)
prediction.show(5)


# In[47]:


# Writing prediction to a csv file on instance
submission=prediction.select("id","prediction")
submission.show(5)
submission = submission.toPandas()

# output the result into a csv file
submission.to_csv('/home/jiawenz1_c4gcp/answer.csv', index=False)


# In[ ]:




