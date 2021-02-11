import sys
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV

from sqlalchemy import create_engine

def load_data(database_filepath):
    """Load data from the input file path."""
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql(database_filepath, engine)
   
    X = df.message
    Y = df.iloc[:, 4:]
    column_names = Y.columns
    
    return X, Y, column_names

def tokenize(text):
    """Tokenize and lemmatize the input text and return the cleaned tokens."""
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    # Remove stop words
    clean_tokens = [w for w in clean_tokens if w not in stopwords.words("english")]
    return clean_tokens


def build_model():
    """Create and return a model with CountVectorizing and TFIDF using SK-Learn's pipeline.
    Returns:
    pipeline: Pipeline model
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])

    parameters = {
        'clf__estimator__n_neighbors': [3,5]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate the model on the given test data.
    Parameters:
    model: Trained ML model
    X_test: Test data
    Y_test: Test target values
    category_names: All possible category names
    """
    y_pred = model.predict(X_test)

    classification_report(Y_test.iloc, y_pred, target_names=category_names)


def save_model(model, model_filepath):
    """Save the model under the specified file path.
    Parameters:
    model: Trained ML model
    model_filepath: Location of the model
    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()