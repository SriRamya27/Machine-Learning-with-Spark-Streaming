'''
python3 stream.py -f sentiment -b 10000 
$SPARK_HOME/bin/spark-submit stream_preprocess.py > output.txt
'''

import numpy as np
import pickle
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score
from sklearn import metrics
#loading all models which were pickled from disk
nb_loaded_model = pickle.load(open('nb_finalized_model.sav', 'rb'))
perceptron_loaded_model = pickle.load(open('perceptron_finalized_model.sav', 'rb'))
sgd_loaded_model = pickle.load(open('sgd_finalized_model.sav', 'rb'))
kmeans_loaded_cluster = pickle.load(open('kmeans_finalized_cluster.sav', 'rb'))


def naiveBayes(x,y):	
	
	nb_loaded_model.partial_fit(x,y, np.unique(y))
	#to predict training data
	print("in nb")
	y_pred=nb_loaded_model.predict(x)
	rmse = np.sqrt(metrics.mean_squared_error(y, y_pred))
	accuracy=accuracy_score(y, y_pred)
	#print(rmse)
	print(accuracy)
	
		
	
def perceptron(x,y):	
	
	perceptron_loaded_model.partial_fit(x,y, np.unique(y))
	print("in prep")
	y_pred=perceptron_loaded_model.predict(x)
	rmse = np.sqrt(metrics.mean_squared_error(y, y_pred))
	accuracy=accuracy_score(y, y_pred)
	print(accuracy)
	#print(rmse)
	


def sdg(x,y):	
	
	sgd_loaded_model.partial_fit(x,y, np.unique(y))
	print("in sdg")
	y_pred=sgd_loaded_model.predict(x)
	rmse = np.sqrt(metrics.mean_squared_error(y, y_pred))
	accuracy=accuracy_score(y, y_pred)
	#print(accuracy)
	print(rmse)
	
	



def kmeans(x,y):	
	
	kmeans_loaded_cluster.partial_fit(x)
	y_pred=kmeans_loaded_cluster.predict(x)
	print("in kmeans")
	rmse = np.sqrt(metrics.mean_squared_error(y, y_pred))
	accuracy=accuracy_score(y, y_pred)
	#print(accuracy)
	print(rmse)
	

#saving all models to disk again to ensure it learns
def save():
	pickle.dump(nb_loaded_model, open('nb_finalized_model.sav', 'wb'))
	pickle.dump(kmeans_loaded_cluster, open('kmeans_finalized_cluster.sav', 'wb'))
	pickle.dump(sgd_loaded_model, open('sgd_finalized_model.sav', 'wb'))
	pickle.dump(perceptron_loaded_model, open('perceptron_finalized_model.sav', 'wb'))

