#This file contains testing functions for each of the models. It also has a function that displays the statistics associated with the tested values.

from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
import numpy as np
import pickle
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score

def testNaiveBayes(x,y):	
	nb_loaded_model = pickle.load(open('nb_finalized_model.sav', 'rb'))
	y_pred= nb_loaded_model.predict(x)
	rmse = np.sqrt(metrics.mean_squared_error(y, y_pred))
	accuracy=accuracy_score(y, y_pred)
	print_evaluate(y, y_pred)
		
def testPerceptron(x,y):	
	perceptron_loaded_model = pickle.load(open('perceptron_finalized_model.sav', 'rb'))
	y_pred= perceptron_loaded_model.predict(x)
	rmse = np.sqrt(metrics.mean_squared_error(y, y_pred))
	accuracy=accuracy_score(y, y_pred)
	print_evaluate(y, y_pred)

def testSdg(x,y):	
	sgd_loaded_model = pickle.load(open('sgd_finalized_model.sav', 'rb'))
	y_pred= sgd_loaded_model.predict(x)
	rmse = np.sqrt(metrics.mean_squared_error(y, y_pred))
	accuracy=accuracy_score(y, y_pred)
	print_evaluate(y, y_pred)

def testKmeans(x,y):	
	kmeans_loaded_cluster = pickle.load(open('kmeans_finalized_cluster.sav', 'rb'))
	y_pred= kmeans_loaded_cluster.predict(x)
	rmse = np.sqrt(metrics.mean_squared_error(y, y_pred))
	accuracy=accuracy_score(y, y_pred)
	print_evaluate(y, y_pred)

def print_evaluate(true, predicted): 
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print(confusion_matrix(true, predicted))
    print(classification_report(true, predicted))
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('______________________')
