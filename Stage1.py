
# coding: utf-8

# # Stage 1: Define Information Need and Evaluate Data

# # Problem Statement
# 
# When customers shop on homedepot.com, they expect to find the right products quickly. If Home Depot can accurately predict the relevance between customers’ search queries and the returned products and then pop out the products that are most relevant to customer’s need, it would attract more customers. Search relevancy is an implicit measure Home Depot utilizes to evaluate how quickly they can get customers to the desired products. Their current approach to evaluate the search algorithms is using human raters to assign a relevance score to the query/product pair. However, this is a slow and subjective process. Moreover, there are infinite search/product pairs, human raters are not able to give a relevance score to all possible query/product pairs. Therefore, Home Depot hopes to build a predictive model which can automatically predicts a relevance score for every pair of search query and its returned product. 
# 
# # Dataset
# 
# Four data sets are provided for by Home Depot. There are 74,067 entries in training data and 166,694 entries in testing data. Each row in the train or test data sets represents a query/product pair, where the product_uid key can be used to join product_description and product_attributes data sets to extract product information including description, bullets, brand, material, color, functionality, etc. In the training data set, each query/product pair is given a relevance score. The relevance score is between 1 (not relevant) to 3 (perfect match). A score of 2 represents partially or somewhat relevant. Each search/product pair was evaluated by several human raters. The provided relevance scores are the average value of their ratings. The metric to evaluate the prediction errors is Root Mean Square Error (RMSE), the RMSE can be 1 at most. In the Kaggle competition, the best RMSE score is 0.43192. But such score was achieved by teams who have ensembled and stacked on thousands of feature columns and different models. In this task, it is impossible for us to obtain similar score. A RMSE score of around 0.50 would be acceptable. 
# 
# # Feature Engineering
# 
# Because the English can be not understandable by the machine learning model, we need to extract numeric features from the text and create new training and testing data sets for modeling.
# 
# The feature engineering is comprised of two parts. The first part is Data Preprocessing and the second part is feature extraction.
# 
# I. Data Preprocessing
# 
#     a. Tokenization
# 
#     For the training dataset and test dataset, we did the following processing:
#     
#     (1) Merge the product description into the train data set on product_id.
#     
#     (2) Extract the “brand” from the product attribute for each product_id.
#     
#     (3) Merge the “brand” attributes into train data set with product_id.
#     
#     (4) Tokenize the search_term, product_title, product_discription, and product_attribute strings 
#     
#     (5) Word stemming in product description.
# 
#     b. Word Vector
#     
#     The search_term, product_title, product_discription, and product_attributes columns in the training and test data sets can be considered as a set of documents, each of which is represented as a sequence of words. We use Word2Vec to transform each document into a feature vector which then can be used for document similarity calculation.
# 
# II. Feature Extraction
# 
#     a. Cosine Similarity between query and product title.
#     
#     b. Cosine Similarity between query and product attribute.
#     
#     c. Eucledian Distance between query and product title.
#     
#     d. Eucledian Distance between query and product attribute.
#     
#     e. Number of matched words between query and product title.
#     
#     f. Number of matched words between query and product description.
#     
#     g. Number of matched words between query and product attribute.
#     
#     h. Length of the Search Term.
# 
# III. Machine Learning Modeling
# 
#     a. Linear Regression Model
#     
#     b. Random Forest Model
# 
# 
# # Kaggle Performace
# 
# 
# 1) Initially, we used only 3 features, namely the cosine similarity between serach and product (attributes, title, description). We received a score of 0.536 on first submission.
# 
# 
# 2) On further exploration, we added the euclidean distance to the vector list too. Now there were 6 features and the performance increased to 0.5181.
# 
# 
# 3) To further improve performance, we added two new featurs matching the number of terms between search terms and product (attributes, title, description). Also, we removed the cosine similarity and euclidean distance between search term and description. This improved the score to 0.4988.
# 
# 
# 4) Finally, we added another feature that contained the size of search term. This helped us boost performance to 0.49618.
# 

# In[ ]:




