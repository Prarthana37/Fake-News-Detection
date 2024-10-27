# Fake-News-Detection
## INRODUCTION:
The spreading of fake news causes many problems in the society. It easily deceives people and leads to confusion among them. It has the ability to cause a lot of social and national damage with destructive impacts. Sometimes it gets very difficult to know if the news is genuine or fake. Therefore it is very necessary to detect if the news is fake or not. Through this project we will be detecting whether the news is fake, unreliable or not by using certain Natural Language Processing Techniques (NLP).
     
## PROJECT OVERVIEW:
- The ultimatum of this project is to accurately distinguish the news available on the internet being genuine and true or being fake and unreliable.
- This will provide a transparent insight about the news in the journals, websites for the readers.
- I have used a technique called Natural Language Processing (NLP) in this attempt.
- A suitable model is also developed for the distinction between the true and the fake news.
- The prominent reason why we have used NLP technique is that because NLP algorithms can ascertain the intention and any biases of an author by analyzing the emotions displayed in the news.

## DESIGN THINKING
The approach to solving this problem can be classified into several phases, each with specific objectives and tasks. This structured approach will ensure that we systematically address all aspects of the problem.
### These phases will include:
1.	Acquiring Data from Data Source
2.	Data preprocessing
3.	Feature Engineering
4.	Text Preprocessing
5.	Feature Extraction
6.	Model Selection
7.	Model Building
8.	Model Fitting 
9.	Model Training
10.	Evaluation 

## LIST OF TOOLS AND SOFTWARE COMMONLY USED IN THE PROCESS:
1.	Programming Language - Python
2.	Integrated Development Environment(IDE) - Jupyter Notebook
3.	Machine Learning Libraries - scikit-learn, Tensorflow, Keras API 
4.	Data Visualization Tools: matplotlib, seaborn, plotly, cufflinks
5.	Data Preprocessing Tools: pandas library,  nltk library

## Step 1- ACQUIRING DATA:
I have obtained the datasets from “Kaggle” that contains the list of articles that are considered to be true and fake. It helps in the distinction between the fake news from the true news.
The link for the dataset is : [https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset]

## Step 2- DATA PRPEPROCESSING:
Cleaning process initially involved the assigning of the labels to Fake and True news datasets.
We have assigned “F” label for the Fake news and “T” label for the True news 
It involves checking for null values, lowercasing all the text, removing all special characters and stopwords.

## Step 3- FEATURE ENGINEERING:
Feature Engineering is the process of taking raw data and transforming them into certain features that help in creating a predictive model using standard modelling methods.
It is also a form of Exploratory Data Analysis.
This includes :
3.1)	Highlighting the features of dataset
3.2)	N-Gram Analysis (Unigram, Bigram)
3.3)	WordCloud

## step 4- TEXT PREPROCESSING:
In the context of LSTM developing the models for natural language processing (NLP) tasks like fake news detection, text preprocessing plays a vital role. 	Then we have performed Stemming. 
Stemming is a method of deriving root word from the inflected word. Here we extract the reviews and convert the words in reviews to its root word. For example, Going -> go, Finally -> fina

## Step 5- FEATURE EXTRACTION:
In the context of developing the models for natural language processing (NLP) tasks like fake news detection, feature extraction primarily involves converting textual data into numerical representations that can be fed into the model. Using train test split function we are splitting the dataset into 80:20 ratio for train and test set respectively 
It involves 3 steps namely:
5.1) Word Embedding by TF-IDF
5.2) One hot Representation
5.3) Padding

## Step 6- MODEL SELECTION:
We should select a suitable classification algorithm (e.g., Logistic Regression, Random Forest, or Neural Networks) for the fake news detection task.
Among the classification algorithms, we have chosen the LSTM (Long Short Term Memory) for building the model.
LSTM (Long Short Term Memory) which helps in containing sequence information.
Long Short-Term Memory (LSTM) networks are a type of recurrent neural network capable of learning order dependence in sequence prediction problems. 

## Step 7- MODEL BUILDING:
At first we are going to develop the base model and compile it. 
First layer will be the embedding layer which has the input of vocabulary size, vector features and sentence length. 
Later we add 30% dropout layer to prevent over-fitting and the LSTM layer which has100 neurons in the layer.
In final layer we use sigmoid activation function. Later we compile the model using adam optimizer and binary cross entropy as loss function since we have only two outputs.

## Step 8- MODEL FITTING:
Before fitting to the model, we considered the padded embedded object as X and y as y itself and have converted them into an array.

## Step 9- MODEL TRAIINING:
We have split our new X and y variable into train and test and proceed with fitting the model to the data. We have considered 10 epochs and 128 as batch size. 
The number of epochs and batch size can be varied to get better results.

## Step 10- MODEL EVALAUTION:
Evaluation metrics are used to measure the quality of the statistical or machine learning model. 
Evaluating machine learning models or algorithms is essential for any project. 
There are many different types of evaluation metrics available to test a model. 
Here we have used:
10.1)	Confusion Matrix - 	It is a table that is used in classification problems to assess where errors in the model were made.
10.2)	Accuracy - Accuracy measures how often the model is correct.
10.3)	Classification Report - the summary of the quality of classification made by the constructed ML model that comprises mainly 5 columns and (N+3) rows. 
10.4)	ROC-AUC Curve - An ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds and AUC stands for "Area under the ROC Curve." That is, AUC measures the entire two-dimensional area underneath the entire ROC curve (think integral calculus) from (0,0) to (1,1).

## CONCLUSION:
In conclusion, the use of Natural Language Processing (NLP) and Long Short-Term Memory (LSTM) networks for fake news detection represents a significant step forward in combating the proliferation of false information in our digital age. Through the application of these advanced technologies, We can achieve more accurate and efficient identification of deceptive content within news articles, social media posts, and various online sources. 
