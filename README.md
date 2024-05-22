OVERVIEW OF THE ANALYSIS
This project aims to develop a predictive model to determine if applicants for aid from the non-profit organization Alphabet Soup will effectively utilize the funding they request. It applies a neural network deep learning model to make binary predictions of funding success, utilizing a sample of more than 34,000 previous aid grants. The data was preprocessed by removing irrelevant columns, binning rare variable values, and converting categorical data to numeric for modeling purposes. TensorFlow Keras was used to build a neural network model, adjusting the number of layers and neurons based on the number of features to achieve accurate results. Since the initial model's accuracy fell below 75%, multiple subsequent attempts were made to optimize the model and improve its accuracy.

VARIABLES that capture metadata about each organization:
EIN and NAME—Identification columns
APPLICATION_TYPE—Alphabet Soup application type
AFFILIATION—Affiliated sector of industry
CLASSIFICATION—Government organization classification
USE_CASE—Use case for funding
ORGANIZATION—Organization type
STATUS—Active status
INCOME_AMT—Income classification
SPECIAL_CONSIDERATIONS—Special considerations for application
ASK_AMT—Funding amount requested
IS_SUCCESSFUL—Was the money used effectively

RESULTS:
- DATA PREPROCESSING
The target variable for our model is IS_SUCCESSFUL.
The feature variables for our model are APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT.
The EIN and NAME columns were removed because they are neither targets nor features.

-COMPILING, TRAINING, and EVALUATING the MODEL
The model was built with 3 hidden layers because there was a high number of input dimensions. In the last optimization model attempt, modification to 2 hidden layers had little impact on the model's accuracy. I used 1 input, 3 hidden layers and 1 output layer, with the first hidden layer having largest number of neurons and decreasing with the addition of more layers. I used a combination of tanh, ReLU and sigmoid as activation functions, however using ReLU once and sigmoid twice for our model produced the highest accuracy of 73.3%. I followed with making additional 3 optimization attempts in order to achieve greater desired accuracy score of 75%. However, they resulted in accuracies of 73.2%, 73.1%, and 73.3% respectively.

SUMMARY:
In short, the model produced the greatest accuracy score of 73% despite running multiple varied optimization methods. Using Random Forest modeling might lead to higher accuracy by reducing overfitting as it will create bootstrap samples from the original dataset.
