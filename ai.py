from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import  ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from textblob import Word
import re
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
import os

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('omw-1.4')

import warnings
warnings.filterwarnings("ignore")

current_directory = os.path.dirname(os.path.realpath(__file__)) # for file operations
models = {'Logistic Regression': None, 'Random Forest' : None, 'Gradient Boosting' : None, 'K-Neighbors' : None, 'MLP' : None}
metrics = {'Logistic Regression': [], 'Random Forest' : [], 'Gradient Boosting' : [], 'K-Neighbors' : [], 'MLP' : []}

def preprocess(dataset = None, flag = 1):
    if flag == 0: # dataset includes all inputs
        df = dataset.copy()
        body = df['Body']

        # Model can be confused due to the links. We will extract and remove links with regex(regular expressions). There is also alternatives like BeatifulSoup(for HTML labels), linkify etc.
        for i in range(len(body)):
                link_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', re.IGNORECASE)
                df.loc[i,"Body"] = re.sub(link_pattern, '', df.loc[i,"Body"])

        # convert upper cases to lower cases:
        df['Body'] = df['Body'].apply(lambda line: " ".join(text.lower() for text in line.split())) 

        # removing punctuations:
        df['Body'] = df['Body'].replace("[^\w\s]","", regex=True)  

        # removing numbers:
        df['Body'] = df['Body'].replace("\d","",regex=True)

        # removing stopwords:
        sw = stopwords.words("english")
        df['Body'] = df['Body'].apply(lambda line: " ".join(text for text in line.split() if text not in sw))

        # to find root of a word, we can use stemming and lemmatization. Stemming is faster but has lower accuracy so we'll use lemmatization
        df['Body'] = df['Body'].apply(lambda line: " ".join([Word(text).lemmatize() for text in line.split()]))

        return df

    else: # if input is just a text
        text = dataset[0]

        link_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', re.IGNORECASE)
        text = re.sub(link_pattern, '', text)

        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d', '', text)
        
        tokenizer = WordPunctTokenizer()
        words = tokenizer.tokenize(text)
        
        sw = stopwords.words("english")
        words = [word for word in words if word not in sw]
        words = [Word(word).lemmatize() for word in words]
        return " ".join(words)
        
def eval_metrics(model, X_test, Y_test):
      predict = model.predict(X_test)
      confusion = confusion_matrix(Y_test,predict)
      accuracy = accuracy_score(Y_test,predict)
      precision = precision_score(Y_test, predict)
      recall = recall_score(Y_test, predict)
      f1 = f1_score(Y_test,predict)
      print("accuracy_score:",str(accuracy),"\n",
            "precision_score:",str(precision),"\n",
            "recall_score:",str(recall),"\n",
            "f1_score:",str(f1),"\n",
            "confusion_matrix:","\n",str(confusion),"\n",
            model.get_params())
      return confusion, accuracy, precision, recall, f1


def writeModels():
    
    # reading input files:
    file_path = os.path.join(current_directory, "enronSpamSubset.csv")
    enronSpam = pd.read_csv(file_path)
    file_path = os.path.join(current_directory, "lingSpam.csv")
    lingSpam = pd.read_csv(file_path)
    file_path = os.path.join(current_directory, "completeSpamAssassin.csv")
    completeSpam = pd.read_csv(file_path)
    
    # data cleaning and merging: 
    completeSpam.dropna(inplace=True)
    enronSpam.drop(["Unnamed: 0.1",	"Unnamed: 0"], axis=1, inplace=True)
    lingSpam.drop(["Unnamed: 0"], axis=1, inplace=True)
    completeSpam.drop(["Unnamed: 0"], axis=1, inplace=True)
    dataset = pd.concat([enronSpam,lingSpam,completeSpam], axis=0, ignore_index=True)
    
    df_preprocessed = preprocess(dataset,0) #preprocessing step
    
    X, Y = df_preprocessed['Body'], df_preprocessed['Label']
    tf_idf_vectorizer = TfidfVectorizer()
    tf_idf_vectorizer.fit(X)
    X= tf_idf_vectorizer.transform(X)
    X_train,X_test,Y_train,Y_test = train_test_split(X, Y, random_state=42) # split datas
    
    # generating models:
    model_logistic = linear_model.LogisticRegression().fit(X_train, Y_train)
    model_rf = ensemble.RandomForestClassifier().fit(X_train, Y_train)
    model_gbm = ensemble.GradientBoostingClassifier().fit(X_train, Y_train)
    model_knn = KNeighborsClassifier(n_neighbors=38).fit(X_train, Y_train) # best n_neighbors value is 38, according to model tuning step below 
    """
    from sklearn.model_selection import GridSearchCV

    model_knn = KNeighborsClassifier().fit(X_train,Y_train)
    knn_params = {
        "n_neighbors": np.arange(1,51)
    }
    knn_cv_model = GridSearchCV(model_knn,knn_params,cv=10,n_jobs=-1,verbose=2).fit(X_train,Y_train)
    best = knn_cv_model.best_params_ # n_neighbors is 38

    #final model:
    knn_tuned_model = KNeighborsClassifier(n_neighbors=best["n_neighbors"]).fit(X_train,Y_train)
    """
    model_mlp = MLPClassifier().fit(X_train, Y_train)
    
    # metric evaluation:
    confusion, accuracy, precision, recall, f1 = eval_metrics(model_logistic, X_test=X_test, Y_test=Y_test)
    metrics['Logistic Regression'] = [confusion, accuracy, precision, recall, f1]

    confusion, accuracy, precision, recall, f1 =  eval_metrics(model_rf, X_test=X_test, Y_test=Y_test)
    metrics['Random Forest'] = [confusion, accuracy, precision, recall, f1]

    confusion, accuracy, precision, recall, f1 = eval_metrics(model_gbm, X_test=X_test, Y_test=Y_test)
    metrics['Gradient Boosting'] = [confusion, accuracy, precision, recall, f1]
    
    confusion, accuracy, precision, recall, f1 = eval_metrics(model_knn, X_test=X_test, Y_test=Y_test)
    metrics['K-Neighbors'] = [confusion, accuracy, precision, recall, f1]
    
    confusion, accuracy, precision, recall, f1 =  eval_metrics(model_mlp, X_test=X_test, Y_test=Y_test)
    metrics['MLP'] = [confusion, accuracy, precision, recall, f1]

    # saving models, metrics and vectorizer into files:
    file_path = os.path.join(current_directory, "model_linear.pkl")
    with open(file_path, 'wb') as file:
        pickle.dump(model_logistic, file)

    file_path = os.path.join(current_directory, "model_knn.pkl")  
    with open(file_path, 'wb') as file:
        pickle.dump(model_knn, file)
        
    file_path = os.path.join(current_directory, "model_gbm.pkl")  
    with open(file_path, 'wb') as file:
        pickle.dump(model_gbm, file)
        
    file_path = os.path.join(current_directory, "model_rf.pkl")  
    with open(file_path, 'wb') as file:
        pickle.dump(model_rf, file)
        
    file_path = os.path.join(current_directory, "model_mlp.pkl")  
    with open(file_path, 'wb') as file:
        pickle.dump(model_mlp, file)
        
    file_path = os.path.join(current_directory, "model_metrics.pkl")  
    with open(file_path, 'wb') as file:
        pickle.dump(metrics, file)
        
    file_path = os.path.join(current_directory, "vectorizer.pkl")  
    with open(file_path, 'wb') as file:
        pickle.dump(tf_idf_vectorizer, file)
        

def readModels():

    # reading files
    file_path = os.path.join(current_directory, "model_linear.pkl")
    with open(file_path, 'rb') as file:
        models['Logistic Regression'] = pickle.load(file)

    file_path = os.path.join(current_directory, "model_rf.pkl")  
    with open(file_path, 'rb') as file:
        models['Random Forest'] = pickle.load(file)

    file_path = os.path.join(current_directory, "model_gbm.pkl")  
    with open(file_path, 'rb') as file:
        models['Gradient Boosting'] = pickle.load(file)

    file_path = os.path.join(current_directory, "model_knn.pkl")  
    with open(file_path, 'rb') as file:
        models['K-Neighbors'] = pickle.load(file)

    file_path = os.path.join(current_directory, "model_mlp.pkl")  
    with open(file_path, 'rb') as file:
        models['MLP'] = pickle.load(file)
        
    file_path = os.path.join(current_directory, "model_metrics.pkl")  
    with open(file_path, 'rb') as file:
        metrics= pickle.load(file)
        
    return models, metrics

def predictMail(text,selectedModel):

    # taking an input text and preprocess:
    comment = pd.Series(text)
    comment = preprocess(comment, 1)

    # vectorization:
    file_path = os.path.join(current_directory, "vectorizer.pkl")  
    with open(file_path, 'rb') as file:
        loaded_vectorizer = pickle.load(file)

    comment = loaded_vectorizer.transform([comment])

    # calculating confidence:
    if hasattr(selectedModel, "predict_proba"):
        probabilities = selectedModel.predict_proba(comment)
        max_probability_index = np.argmax(probabilities)
        predicted_class = selectedModel.classes_[max_probability_index]
        confidence = probabilities[0, max_probability_index]
        return predicted_class, confidence
    else:
        predicted_class = selectedModel.predict(comment)
        confidence = None
        return predicted_class, confidence
    
# compares models' metrics and visualizes:
def showAnalysis(metrics, metricName, models, index, color):
    results = pd.DataFrame(columns = ["Models",metricName])
    for model_name, _ in models.items():
        print(metrics[model_name][index])
        results.loc[len(results)] = [model_name, metrics[model_name][index]]
        
    sns.barplot(x= metricName, y = "Models", data = results, color = color)
    plt.xlabel(metricName)
    plt.title("Model's " +  metricName  + " Scores")
    plt.show()
    
# finds most frequent words in dataset and visualizes:
def showFreqs(df_preprocessed):
    terms = pd.Series(" ".join(df_preprocessed['Body']).split()).value_counts().reset_index()
    terms.columns = ["word","frequency"]
    frequents = terms[(terms['frequency'] > 10000) & (terms['frequency'] < 50000)]
    print(frequents)
    frequents.plot.bar(x = 'word', y = 'frequency');