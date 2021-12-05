'''
python3 stream.py -f sentiment -b 10000 
$SPARK_HOME/bin/spark-submit stream_preprocess.py > output.txt
'''

import numpy as np
import pickle
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import MiniBatchKMeans, KMeans
nb_loaded_model = pickle.load(open('nb_finalized_model.sav', 'rb'))
perceptron_loaded_model = pickle.load(open('perceptron_finalized_model.sav', 'rb'))
sgd_loaded_model = pickle.load(open('sgd_finalized_model.sav', 'rb'))
kmeans_loaded_cluster = pickle.load(open('kmeans_finalized_cluster.sav', 'rb'))
def naiveBayes(x,y):	
	# load the model from disk
	
	nb_loaded_model.partial_fit(x,y, np.unique(y))
	print("in nb")
	print(nb_loaded_model.predict(x[2]),y[2])
	
#naiveBayes()	
	
def perceptron(x,y):	
	# load the model from disk
	
	perceptron_loaded_model.partial_fit(x,y, np.unique(y))
	print("in perceptron")
	print(perceptron_loaded_model.predict(x[2]),y[2])
	
#perceptron()

def sdg(x,y):	
	# load the model from disk
	
	sgd_loaded_model.partial_fit(x,y, np.unique(y))
	print("in sgd")
	print(sgd_loaded_model.predict(x[2]),y[2])
	
#sgd()	


def kmeans(x,y):	
	# load the cluster from disk
	
	kmeans_loaded_cluster.partial_fit(x)
	print("in kmeans")
	print(kmeans_loaded_cluster.predict(x[2]),y[2])
	
#kmeans()
def save():
	pickle.dump(nb_loaded_model, open('nb_finalized_model.sav', 'wb'))
	pickle.dump(kmeans_loaded_cluster, open('kmeans_finalized_cluster.sav', 'wb'))
	pickle.dump(sgd_loaded_model, open('sgd_finalized_model.sav', 'wb'))
	pickle.dump(perceptron_loaded_model, open('perceptron_finalized_model.sav', 'wb'))

