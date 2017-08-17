
# coding: utf-8

# # Stage 2: Building minimum pipeline
# 
# ### Approach: 
# 
# Based on the results of stage 1, we first extracted brand, color, and material information from the product attributes file, and merged them as well as the product description onto the training data file. 
# 
# Subsequently, we split text data in the training set into array of words under each column, and used Word2Vec as the first transformer to distribute each data set into different, continuous dimensions. 
# 
# Next, we created three features derived from cosine similarities between Search Term and Product Title, Product Description and Product Attributes, respectively. 
# 
# At last, we assembled the three features and put them into Random Forest Regression estimator to build our first experimental pipeline, which was later used on testing data to predict the relevancy score of given search term and product in the test data file.
# 

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


# ###  Processing training data

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


# ### Processing testing data

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


# ### Building pipeline using assembled 3 features and RandomForestRegressor as estimator, train it with training data, and use it on testing data.

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


# ### Observation and Analysis: 
# 
# All predictions from this pipeline landed in the range of 2.30 - 2.46, this biased prediction might have resulted from:
# 
# (1)uneven length of text under each column. For example, the product description is much longer than extracted attributes or product title, the data intensity is much lower in  the product description after it is distributed into 10 dimensions than that of attribues and product title;
# 
# (2)cosine similarity can only convey the angles between objectives and features, but not the distance between features and objectives in the 10 dimension space;
# 
# (3) We didn't choose the optimum number of dimension for each variables when using Word2Vec

# # Stage 4: Improve transformers
# 
# Based on the observation in stage 2, we took 3 actions to improve our transformers:
# 
# (1) We added two new features. The Euclidean distance between Search Title and Product (Title and Attributes) was used along with Word2Vec so as to take distance information into the model;
# 
# (2) We added Word match function as transformer to obtain absolute values to get rid off the issue about the different length of product_description, product_title, and product_attribues.
# 
# (3) We added a feature for the length of the Search Term. 
# 
# In the end we obtained 8 features and trained a new model with them.

# In[ ]:


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


# In[ ]:


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


# ### Processing training data

# In[ ]:


# call the idfUDF function to calculate the Cosine Similarity between each two vectors
training = training.withColumn("searchAndTitle", idfUDF("search_term_array_features","product_title_array_features"))
training = training.withColumn("searchAndAttributes", idfUDF("search_term_array_features","product_attributes_array_features"))

# call the edUDF function to calculate the Eucledian Distance between each two vectors
training = training.withColumn("searchAndTitleED", edUDF("search_term_array_features","product_title_array_features"))
training = training.withColumn("searchAndAttributesED", edUDF("search_term_array_features","product_attributes_array_features"))

# call the matchUDF function to calculate the number of matched words between each two vectors
training = training.withColumn("searchAndTitleMD", matchUDF("search_term_","product_title_"))
training = training.withColumn("searchAndAttributesMD", matchUDF("search_term_","product_attributes_"))
training = training.withColumn("searchAndDescriptionMD", matchUDF("search_term_","product_description_"))

# combine three features into a dictionary
features=["searchAndTitleMD", "searchAndAttributesMD", "searchAndDescriptionMD", "search_term_length", "searchAndTitleED", "searchAndAttributesED", "searchAndTitle", "searchAndAttributes"]
assembler_features = VectorAssembler(inputCols=features, outputCol='features')
training = assembler_features.transform(training)
training.select("features").show(10)


# In[ ]:


+--------------------+
|            features|
+--------------------+
|[1.0,0.0,10.0,2.0...|
|[1.0,0.0,5.0,3.0,...|
|[1.0,0.0,3.0,1.0,...|
|[1.0,0.0,14.0,2.0...|
|[5.0,1.0,26.0,3.0...|
|[1.0,0.0,15.0,3.0...|
|[4.0,0.0,4.0,2.0,...|
|[4.0,0.0,8.0,3.0,...|
|[0.0,0.0,0.0,1.0,...|
|[0.0,0.0,0.0,2.0,...|
+--------------------+
only showing top 10 rows


# ### Processing testing data

# In[ ]:


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


# In[ ]:


+---+-----------+---------+-------------------+--------------------+---------------------+--------------------+--------------------+--------------------------+----------------------------+---------------------------------+--------------------+--------------------+------------------+---------------------+----------------+---------------------+----------------------+--------------------+
| id|product_uid| term_len|       search_term_|      product_title_|product_description__|product_description_| product_attributes_|search_term_array_features|product_title_array_features|product_attributes_array_features|      searchAndTitle| searchAndAttributes|  searchAndTitleED|searchAndAttributesED|searchAndTitleMD|searchAndAttributesMD|searchAndDescriptionMD|            features|
+---+-----------+---------+-------------------+--------------------+---------------------+--------------------+--------------------+--------------------------+----------------------------+---------------------------------+--------------------+--------------------+------------------+---------------------+----------------+---------------------+----------------------+--------------------+
| 28|     100010|        2|   [anchor, stakes]|[valley, view, in...| [valley, view, in...|[valley, view, in...|[valley, view, in...|      [-0.3361585959792...|        [-0.0999963407715...|             [0.05088427911202...|-0.06568813922896556|  -0.445601333615832|0.8926519234241485|     1.10827180560135|               2|                    0|                    20|[2.0,0.0,20.0,2.0...|
| 29|     100010|        2|[landscape, edging]|[valley, view, in...| [valley, view, in...|[valley, view, in...|[valley, view, in...|      [-0.3577062934637...|        [-0.0999963407715...|             [0.05088427911202...| -0.2229992001177521| 0.04072888497063595| 1.152480939061946|    1.082240603864786|               0|                    0|                     7|[0.0,0.0,7.0,2.0,...|
| 30|     100010|        2|     [lawn, edging]|[valley, view, in...| [valley, view, in...|[valley, view, in...|[valley, view, in...|      [0.04160743951797...|        [-0.0999963407715...|             [0.05088427911202...|-0.31502514232395173|-0.28670860848917473|1.5672070319158358|    1.612794519813814|               0|                    0|                    11|[0.0,0.0,11.0,2.0...|
| 31|     100010|        2|    [metal, stakes]|[valley, view, in...| [valley, view, in...|[valley, view, in...|[valley, view, in...|      [-0.2837830856442...|        [-0.0999963407715...|             [0.05088427911202...|-0.11440542653458408|-0.46263787482356594|0.9505944856778183|   1.1626233042051979|               4|                    0|                    16|[4.0,0.0,16.0,2.0...|
| 32|     100010|        1|           [stakes]|[valley, view, in...| [valley, view, in...|[valley, view, in...|[valley, view, in...|      [-0.4358726441860...|        [-0.0999963407715...|             [0.05088427911202...|-0.02373848211508...| -0.2653196363894832| 1.046374075807638|    1.233656754038182|               2|                    0|                    14|[2.0,0.0,14.0,1.0...|
+---+-----------+---------+-------------------+--------------------+---------------------+--------------------+--------------------+--------------------------+----------------------------+---------------------------------+--------------------+--------------------+------------------+---------------------+----------------+---------------------+----------------------+--------------------+
only showing top 5 rows

+--------------------+
|            features|
+--------------------+
|[2.0,0.0,20.0,2.0...|
|[0.0,0.0,7.0,2.0,...|
|[0.0,0.0,11.0,2.0...|
|[4.0,0.0,16.0,2.0...|
|[2.0,0.0,14.0,1.0...|
|[0.0,0.0,11.0,3.0...|
|[4.0,0.0,7.0,2.0,...|
|[4.0,2.0,6.0,2.0,...|
|[6.0,2.0,19.0,5.0...|
|[3.0,2.0,4.0,3.0,...|
+--------------------+
only showing top 10 rows


# ### Building pipeline using assembled 8 features and RandomForestRegressor as estimator, train it with training data, and use it on testing data.

# In[ ]:


# Using thr Random Forest Regressor on the data
rf = RandomForestRegressor(featuresCol="features",labelCol='relevance', numTrees=200, maxDepth=10)
pipeline = Pipeline(stages=[rf])

# train the model
model = pipeline.fit(training)

# testing
prediction = model.transform(testing)
prediction.show(5)


# In[ ]:


+---+-----------+---------+-------------------+--------------------+---------------------+--------------------+--------------------+--------------------------+----------------------------+---------------------------------+--------------------+--------------------+------------------+---------------------+----------------+---------------------+----------------------+--------------------+------------------+
| id|product_uid| term_len|       search_term_|      product_title_|product_description__|product_description_| product_attributes_|search_term_array_features|product_title_array_features|product_attributes_array_features|      searchAndTitle| searchAndAttributes|  searchAndTitleED|searchAndAttributesED|searchAndTitleMD|searchAndAttributesMD|searchAndDescriptionMD|            features|        prediction|
+---+-----------+---------+-------------------+--------------------+---------------------+--------------------+--------------------+--------------------------+----------------------------+---------------------------------+--------------------+--------------------+------------------+---------------------+----------------+---------------------+----------------------+--------------------+------------------+
| 28|     100010|        2|   [anchor, stakes]|[valley, view, in...| [valley, view, in...|[valley, view, in...|[valley, view, in...|      [-0.3361585959792...|        [-0.0999963407715...|             [0.05088427911202...|-0.06568813922896556|  -0.445601333615832|0.8926519234241485|     1.10827180560135|               2|                    0|                    20|[2.0,0.0,20.0,2.0...| 2.446440522970927|
| 29|     100010|        2|[landscape, edging]|[valley, view, in...| [valley, view, in...|[valley, view, in...|[valley, view, in...|      [-0.3577062934637...|        [-0.0999963407715...|             [0.05088427911202...| -0.2229992001177521| 0.04072888497063595| 1.152480939061946|    1.082240603864786|               0|                    0|                     7|[0.0,0.0,7.0,2.0,...|2.2576765608191876|
| 30|     100010|        2|     [lawn, edging]|[valley, view, in...| [valley, view, in...|[valley, view, in...|[valley, view, in...|      [0.04160743951797...|        [-0.0999963407715...|             [0.05088427911202...|-0.31502514232395173|-0.28670860848917473|1.5672070319158358|    1.612794519813814|               0|                    0|                    11|[0.0,0.0,11.0,2.0...|2.2094052463612033|
| 31|     100010|        2|    [metal, stakes]|[valley, view, in...| [valley, view, in...|[valley, view, in...|[valley, view, in...|      [-0.2837830856442...|        [-0.0999963407715...|             [0.05088427911202...|-0.11440542653458408|-0.46263787482356594|0.9505944856778183|   1.1626233042051979|               4|                    0|                    16|[4.0,0.0,16.0,2.0...| 2.596356535828383|
| 32|     100010|        1|           [stakes]|[valley, view, in...| [valley, view, in...|[valley, view, in...|[valley, view, in...|      [-0.4358726441860...|        [-0.0999963407715...|             [0.05088427911202...|-0.02373848211508...| -0.2653196363894832| 1.046374075807638|    1.233656754038182|               2|                    0|                    14|[2.0,0.0,14.0,1.0...| 2.416165972511263|
+---+-----------+---------+-------------------+--------------------+---------------------+--------------------+--------------------+--------------------------+----------------------------+---------------------------------+--------------------+--------------------+------------------+---------------------+----------------+---------------------+----------------------+--------------------+------------------+
only showing top 5 rows


# In[ ]:


# Writing prediction to a csv file on instance
submission=prediction.select("id","prediction")
submission.show(100)
submission = submission.toPandas()

# output the result into a csv file
submission.to_csv('/home/jiawenz1_c4gcp/answer.csv', index=False)


# In[ ]:


+-----+------------------+
|   id|        prediction|
+-----+------------------+
|   28| 2.446440522970927|
|   29|2.2576765608191876|
|   30|2.2094052463612033|
|   31| 2.596356535828383|
|   32| 2.416165972511263|
|   33|2.1497855947917928|
|  805|2.6523109773878613|
|  806|2.7146741866063677|
|  807| 2.393296902366696|
|  808|2.3984818649067052|
|  809| 2.118347840605101|
|  810|2.5253150904579367|
|  813|2.4211301143614588|
| 1288| 2.572724034778235|
| 1290|2.3496587593892255|
| 1292|2.1523488045647166|
| 1293| 2.368786084370306|
| 1466|2.1416661903973453|
| 1467|2.3676815397661017|
| 1469|2.3351342983369703|
| 1470| 2.625548598565183|
| 1474| 2.153224153610679|
| 1808|2.3538929916766813|
| 1809|2.3600301558952497|
| 1810|2.4057811480076943|
| 1812|2.3598522311670873|
| 1813|2.7131753119763933|
| 1814| 2.351239522654401|
| 1817|  2.25213674469827|
| 1819| 2.271579411528764|
| 1820|2.4394630723378783|
| 3152|2.4254604006642553|
| 3153|  2.44808482225246|
| 3154| 2.286334311498807|
| 3155| 2.656934301091548|
| 3156| 2.280629919624711|
| 4055|2.4631513610864215|
| 4057| 2.458625431743894|
| 4290|2.4398175467694156|
| 4291|2.2401144172522804|
| 4292|2.5419472292174685|
| 4293|2.2649321616744493|
| 4294| 2.589738239352313|
| 4295|2.5632321727916305|
| 4297|2.1006639118593196|
| 4298|2.2355856362100837|
| 4462|2.0552508003441683|
| 4463|2.4820912241837947|
| 4464| 2.454976705792156|
| 4465| 2.402918679844081|
| 4466|2.1772049220378635|
| 5579| 2.593365818029706|
| 5580|2.2376585006318166|
| 5581|2.3906152590327867|
| 5583|2.0338092019731757|
| 5584|2.0514608262599014|
| 5586| 2.197788346394772|
| 5890|  2.47529358427655|
| 5891|2.4711042250873274|
| 7391|2.4106504202023036|
| 7392|2.2286799946396867|
| 7393| 2.409018658183868|
| 7394| 2.262525738661433|
| 7395|2.3324810043896194|
| 7396|2.6179119907864505|
| 7398|2.6644523159448266|
| 7402|2.6578212313931675|
| 7403|2.5945784153255023|
| 7404| 2.593617625011017|
| 7405|2.6371707747125464|
| 7462|2.2282855512469673|
| 7464|2.3956886245657527|
| 7465|  2.05913892790408|
| 7467|2.1695946899539367|
| 7468|2.6017874379882837|
| 7469|2.4134612159899604|
| 7471|2.6005039580811546|
| 7472|2.1699730556921146|
| 7473| 2.356660360337005|
| 7475|2.0549301944800815|
| 7476| 2.472234994889206|
| 7477|  2.49083963643339|
| 7478| 2.651372277688535|
| 7479|2.4673338286697337|
| 7481|2.4429731453946446|
| 7482| 2.633614204811696|
| 7484|2.4579979960963767|
| 7485|2.2694426441384903|
| 7487|2.5330192086166123|
| 7488| 2.355339190132791|
| 7489| 2.487114663973176|
| 7491| 2.610149067434229|
|12055|2.2076067017017933|
|12056| 2.529381527606393|
|12058|2.4640897014461474|
|12059|2.4334228340378026|
|12061|2.1063081328081386|
|12062|  2.59455692896032|
|12063|  2.20627199886023|
|12064| 2.055801036370751|
+-----+------------------+
only showing top 100 rows


# # Stage 5: Expand on models
# 
# After previous optimizations, we have achieved reasonably good result. We wanted to experiment on Linear regression to see if we can achieve better scores

# In[ ]:


# Using the Linear Regression on the data
lr = LinearRegression(maxIter=100, regParam=0.3, elasticNetParam=0.8, labelCol='relevance')
pipeline = Pipeline(stages=[lr])

# train the model
model = pipeline.fit(training)

# testing
prediction = model.transform(testing)
prediction.show(5)


# In[ ]:


# Writing prediction to a csv file on instance
submission=prediction.select("id","prediction")
submission.show(100)
submission = submission.toPandas()

# output the result into a csv file
submission.to_csv('/home/jiawenz1_c4gcp/answer.csv', index=False)


# In[ ]:


+---+------------------+
| id|        prediction|
+---+------------------+
|  1|2.2531105570770817|
|  4| 2.516205260017714|
|  5| 2.412751772787061|
|  6|2.1677275793650796|
|  7| 2.564859307760412|
|  8| 2.212191500385579|
| 10|2.6471629641729106|
| 11| 2.492132776452756|
| 12| 2.482303198701883|
| 13|2.6193777209717926|
| 14| 2.711583790979911|
| 15|  2.57199076228244|
| 19|1.9151565923207223|
| 22|2.6101884256926033|
| 24|2.4518054676506185|
| 25| 2.436263331926154|
| 26| 2.113331976576055|
| 28|2.4895287863373605|
| 29| 2.206066819693464|
| 30|2.3932106557464983|
| 31|2.5763032222239843|
| 32|2.4656891664266207|
| 33| 2.466463631936975|
| 36| 2.424626153453097|
| 39| 2.250550207324413|
| 40|1.7955178571428572|
| 41| 2.473871612864385|
| 42|2.4394108173186435|
| 43| 2.371889957996507|
| 44|2.5031172417230647|
| 45|2.7344864137339733|
| 46|2.6756648451065232|
| 47| 1.764080357142857|
| 49|2.6200956479104724|
| 50|2.5818523734218033|
| 52| 2.556875901291691|
| 53| 2.405162823234474|
| 54| 2.159102978033318|
| 55| 2.606694042533913|
| 56| 2.436363143747295|
| 57|2.4425536634053087|
| 58|  2.48120410904154|
| 59|2.4781460623997367|
| 60|2.0468611119144016|
| 61|2.5714128027112695|
| 62|2.5315011553963203|
| 63| 2.605281762527837|
| 64|2.4644916643939885|
| 66|2.5766399940083162|
| 67|2.4260711637974204|
| 68| 2.637975941692466|
| 70|2.6068759416924663|
| 71|2.4218581256005125|
| 72|2.2476848370927316|
| 73| 2.226784837092732|
| 74|2.2476848370927316|
| 76| 2.285351503759398|
| 77|2.2476848370927316|
| 78|2.3152618874835857|
| 79| 2.655402412280702|
| 80|  2.45280272556391|
| 82|2.2786070593149543|
| 83|2.4218581256005125|
| 84| 2.410501254975675|
| 86|2.4180532296821817|
| 87|2.4639621896924844|
| 89|1.5790178571428573|
| 91| 2.241784433514889|
| 93| 2.414751582391984|
| 94|  2.42480272556391|
| 95|  2.45280272556391|
| 96|2.2662408521303257|
| 97|2.2449196982841717|
| 98|2.3914245293728724|
| 99| 2.095840852130326|
|100|2.4732351518926583|
|102|2.4298898488131924|
|103| 2.141519698284172|
|104|2.2781756193368037|
|107|2.2584320593149547|
|108|2.3160756193368037|
|109| 2.577900775258216|
|110|2.4435420585824597|
|111|2.4333337252491267|
|112|2.5939898939890984|
|115| 2.234850097162551|
|116|2.6904339203559866|
|118|2.3391990738894095|
|119| 2.678346858232286|
|121| 2.524744201229788|
|124|2.6193961504589724|
|126|2.5424690087689634|
|128|2.4000426392578142|
|129|2.5503720494153717|
|130|2.3557056159118197|
|131|  2.26387079557869|
|132|2.3263606677033977|
|133| 2.491199104760635|
|134|2.5995409746962594|
|135|2.6029163445094663|
+---+------------------+


# # Stage 6: Optimize pipline programmatically
# 
# After we defined the pipeline, we used paramGrid in PySpark to find the optimal parameters for our model programmatically. Due to computational limit of our computer, we did not choose many values for each parameters. But with the ranges we experimented, the result didn't show any apparent improvement to the model after adding the paraGrid.
# 
# We tried playing around with depth of the trees with the values 5 and 10. We also tried to vary the number of trees between 40, 100 and 200. For the transformer, we tried to change the feature vectures using two totally different feature lists.

# In[ ]:


# Dividing training set and test set in 7:3
(trainingData, testData) = training.randomSplit([0.7, 0.3])

# Define an assembler
featureList1=["searchAndTitleMD", "searchAndAttributesMD", "searchAndDescriptionMD", "search_term_length"]
featureList2=["searchAndTitleMD", "searchAndTitleED", "searchAndAttributesED", "searchAndTitle", "searchAndAttributes"]
features=["searchAndTitleMD", "searchAndAttributesMD", "searchAndDescriptionMD", "search_term_length", "searchAndTitleED", "searchAndAttributesED", "searchAndTitle", "searchAndAttributes"]
assembler_features = VectorAssembler(inputCols=features, outputCol='features')

# Define a random forest regresser
rf = RandomForestRegressor(featuresCol="features",numTrees=15, maxDepth=6, seed=0,labelCol="relevance")

# Define a pipeline containing assembler, indexer and rf
pipeline = Pipeline(stages=[assembler_features, rf])

# Use paramGrid to modify parameters
paramGrid = ParamGridBuilder().addGrid(rf.maxDepth, [5, 10]).addGrid(rf.numTrees, [40, 100, 200]).addGrid(assembler_features.inputCols, [featureList1, featureList2]).build()


# # Stage7: Error analysis and iteration
# 
# During the development of the model, we also used cross-validation and RMSE to measure the performance of our model.

# In[ ]:


# Define CrossValidator()
crossValidation = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(),
                          numFolds=2)

# Drop Old features so that the old featurs do not give an error
trainingData = trainingData.drop("features")
testData = testData.drop("features")
trainingData = trainingData.withColumn('label', functions.column('relevance'))
testData = testData.withColumn('label', functions.column('relevance'))

# Train with new data
model = crossValidation.fit(trainingData)

# Test with new data
predictionOfCrossValidation = model.transform(testData)

# Evaluate the error in cross validation
evaluator = RegressionEvaluator(labelCol="relevance", predictionCol="prediction", metricName="rmse")
RMSE = evaluator.evaluate(predictionOfCrossValidation)
print("The root mean square error after applying paramGrid on training set is %g" % RMSE)


# In[ ]:


The root mean square error after applying paramGrid on training set is 0.494131


# ### Result and Conclusion
# 
# Throughout the process of development we found that the up-stream of a model(i.e. features, transformers) affect the result more significantly than down-stream(i.e. estimator, paraGrid). Appropriate data selection and data preprocessing based on thorough analysis of the problem forms a base to the success of the model.
# 
# Apart from that, choosing different combination of features with the transformer, different transformers, different transformers with estimators, gave us a lot of insight into how the results should be tuned to perfection. It requires good understanding of all kinds of transformers and estimators, and the number of combinations is pretty large.
# 
# For the best results that we achieved, we used a stopWordRemover and word stemming as the data preprocessing methods; the Cosine Similarity between search term and the product title or the product attributes, the Euclidean distance between search term and the product title or the product attributes, and search term frequency in description, title and attributes as features; and Random forest as estimator. The RMSE of this model is about 0.494131.
# 
# 
# ### Future Scope
# We learnt that given more time, we should be able to improve our model. This can be done by extracting more valuable data in attributes, populating the training data with the synonyms of the search term, fixing typos in the data etc. One very promising method would be to use inverse Levenshtein Distance.
# 
# ** Inverse Levenshtein Distance**
# Levenshtein Distance is the measure of the similarity between two strings. The inverse should be a good measure of difference between the two strings. Since the best result that we obtained was using word matching(after stemming), by tweaking the Levenshtein Distance to measure change, we can improve our feature vectors.

# In[ ]:




