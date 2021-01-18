import copy

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


def get_sentiment(prediction):
    if prediction == 1:
        sentiment = "Very Negative"
    elif prediction == 2:
        sentiment = "Negative"
    elif prediction == 3:
        sentiment = "Neutral"
    elif prediction == 4:
        sentiment = "Positive"
    elif prediction == 5:
        sentiment = "Very Positive"
    return sentiment


def run_k_nearest_neighbors(dic_):
    # Read file into dataframe
    file = pd.DataFrame(dic_)
    file.to_csv('sentiments.csv')
    df_og = pd.read_csv("sentiments.csv" ,  index_col=False)


    # generating one row
    df = df_og.sample(n=len(df_og))
    df = df.reset_index(drop=True)

    # Create a label encoder
    label_encoder = preprocessing.LabelEncoder()

    # Convert the string labels into numbers
    # df = df.apply(lambda x: label_encoder.fit_transform(x))

    # Choose the independent and dependent variables
    X = df['Comment']
    y = df['Rating']

    # Use CountVectorizer to convert text into tokens/features
    vect = CountVectorizer(stop_words='english', ngram_range=(1, 1), max_df=.80, min_df=4)

    # Split the dataset in training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Use training data to transform text into counts of features for each message
    vect.fit(X_train)
    X_train_dtm = vect.transform(X_train)
    X_test_dtm = vect.transform(X_test)

    # Create an error list
    error = []

    # Calculate the error rate for K values between 1 and 25
    for i in range(1, 25):

        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train_dtm, y_train)
        pred_i = knn.predict(X_test_dtm)
        error.append(np.mean(pred_i != y_test))

    # Create a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)

    # Train the model 
    knn.fit(X_train_dtm, y_train)

    # Make prediction
    y_train_pred = knn.predict(X_train_dtm)
    y_test_pred = knn.predict(X_test_dtm)

    # Make and print a confusion matrix
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)
    # print("Confusion Matrix of training set: ", cm_train)
    # print("Confusion Matrix of testing set: ", cm_test)
    train_acc = accuracy_score(y_train, y_train_pred) * 100
    test_acc = accuracy_score(y_test, y_test_pred) * 100
    # Print the accuracy
    # print("K-NN model accuracy(in %) of training set:", train_acc)
    # print("K-NN model accuracy(in %) of test set:", test_acc)

    # Print the classification report
    report = classification_report(y_test, y_test_pred)
    # print(report)

    # Use training data to transform text into counts of features for each message
    trainingVector = CountVectorizer(stop_words='english', ngram_range=(1, 1), max_df=.80, min_df=5)
    trainingVector.fit(X)
    X_dtm = trainingVector.transform(X)
    knn_complete = KNeighborsClassifier(n_neighbors=5)
    knn_complete.fit(X_dtm, y)
    #LR_complete = LogisticRegression(solver='lbfgs', max_iter=1000)
    #LR_complete.fit(X_dtm, y)
	
    # Create tags
    # tags = ['Positive', 'Neutral', 'Negative']
    tags = ['Very Negative', 'Negative', 'Very Positive', 'Positive', 'Neutral']

    # Get 5 random reviews
    # sample_data = X_test.sample(n=40, random_state=1)

    # Test the algorith using the test set
    # for review in X_test[:40]:
    i = 0
    cpy_list = []
    for li in dic_:
        d2 = copy.deepcopy(li)
        cpy_list.append(d2)
    for review in df_og['Comment']:
        review = [review]
        test_dtm = trainingVector.transform(review)
        pred_label = knn_complete.predict(test_dtm)
        cpy_list[i]['Sentiment'] = get_sentiment(pred_label[0])
        # print(review, " is predicted ", get_sentiment(pred_label[0]))
        i += 1
    return cpy_list, train_acc, test_acc, report
