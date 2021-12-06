'''
python3 stream.py -f sentiment -b 10000 
$SPARK_HOME/bin/spark-submit stream_preprocess.py > output_classification.txt
'''
import csv
import numpy as np
import pickle
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import accuracy_score
from sklearn import metrics

nb_loaded_model = pickle.load(open('nb_finalized_model.sav', 'rb'))
perceptron_loaded_model = pickle.load(open('perceptron_finalized_model.sav', 'rb'))
sgd_loaded_model = pickle.load(open('sgd_finalized_model.sav', 'rb'))
kmeans_loaded_cluster = pickle.load(open('kmeans_finalized_cluster.sav', 'rb'))

#nb_train_accuracy = open('nb_train_accuracy.txt', 'w')

#fields = ['accuracy', 'error']
#nb_train_txt.write(str(fields))

def naiveBayes(x,y):	
	# load the model from disk
	
	nb_loaded_model.partial_fit(x,y, np.unique(y))
	#print("in nb")
	y_pred=nb_loaded_model.predict(x)
	
	rmse = np.sqrt(metrics.mean_squared_error(y, y_pred))
	accuracy=accuracy_score(y, y_pred)
	rows=[str(accuracy),str(rmse)]
	#print(accuracy)
	#print(rmse)
	
	
#naiveBayes()	
	
def perceptron(x,y):	
	# load the model from disk
	
	perceptron_loaded_model.partial_fit(x,y, np.unique(y))
	#print("in perceptron")
	y_pred=perceptron_loaded_model.predict(x)
	
	rmse = np.sqrt(metrics.mean_squared_error(y, y_pred))
	accuracy=accuracy_score(y, y_pred)
	rows=[str(accuracy),str(rmse)]
	#print(accuracy)
	#print(rmse)
	
	
	
#perceptron()

def sdg(x,y):	
	# load the model from disk
	
	sgd_loaded_model.partial_fit(x,y, np.unique(y))
	#print("in sgd")
	y_pred=sgd_loaded_model.predict(x)
	
	rmse = np.sqrt(metrics.mean_squared_error(y, y_pred))
	accuracy=accuracy_score(y, y_pred)
	rows=[str(accuracy),str(rmse)]
	print(accuracy)
	#print(rmse)
	
#sgd()	


def kmeans(x,y):	
	# load the cluster from disk
	
	kmeans_loaded_cluster.partial_fit(x)
	#print("in kmeans")
	y_pred=kmeans_loaded_model.predict(x)
	
	rmse = np.sqrt(metrics.mean_squared_error(y, y_pred))
	accuracy=accuracy_score(y, y_pred)
	rows=[str(accuracy),str(rmse)]
	print(accuracy)
	#print(rmse)
	
#kmeans()
def save():
	pickle.dump(nb_loaded_model, open('nb_finalized_model.sav', 'wb'))
	pickle.dump(kmeans_loaded_cluster, open('kmeans_finalized_cluster.sav', 'wb'))
	pickle.dump(sgd_loaded_model, open('sgd_finalized_model.sav', 'wb'))
	pickle.dump(perceptron_loaded_model, open('perceptron_finalized_model.sav', 'wb'))

