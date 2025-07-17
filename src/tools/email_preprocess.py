#!/usr/bin/python3

import joblib
import numpy
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif


def preprocess():
    """ 
    This function takes a pre-made list of email texts (word_data.pkl)
    and the corresponding authors (email_authors.pkl) and performs
    a number of preprocessing steps:
        -- splits into training/testing sets (10% testing)
        -- vectorizes into tfidf matrix
        -- selects/keeps most helpful features

    After this, the features and labels are put into numpy arrays, which play nice with sklearn functions.

    Returns:
        features_train, features_test, labels_train, labels_test
    """

    base_dir = os.path.dirname(__file__)
    words_file = os.path.join(base_dir, "word_data.pkl")
    authors_file = os.path.join(base_dir, "email_authors.pkl")

    with open(authors_file, "rb") as authors_file_handler:
        authors = joblib.load(authors_file_handler)

    with open(words_file, "rb") as words_file_handler:
        word_data = joblib.load(words_file_handler)

    features_train, features_test, labels_train, labels_test = train_test_split(
        word_data, authors, test_size=0.1, random_state=42
    )

    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
    features_train_transformed = vectorizer.fit_transform(features_train)
    features_test_transformed = vectorizer.transform(features_test)

    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(features_train_transformed, labels_train)
    features_train_transformed = selector.transform(features_train_transformed).toarray()
    features_test_transformed = selector.transform(features_test_transformed).toarray()

    print("No. of Chris training emails : ", sum(labels_train))
    print("No. of Sara training emails : ", len(labels_train) - sum(labels_train))
    
    return features_train_transformed, features_test_transformed, labels_train, labels_test
