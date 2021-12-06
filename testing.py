'''
python3 stream.py -f sentiment -b 10000 
$SPARK_HOME/bin/spark-submit stream_preprocess.py > output.txt
'''
	

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
	#print(accuracy)
	#print(rmse)
	print_evaluate(y, y_pred)
		
def testPerceptron(x,y):	
	# load the model from disk
	perceptron_loaded_model = pickle.load(open('perceptron_finalized_model.sav', 'rb'))
	
	y_pred= perceptron_loaded_model.predict(x)
	rmse = np.sqrt(metrics.mean_squared_error(y, y_pred))
	accuracy=accuracy_score(y, y_pred)
	#print(rmse)
	#print(accuracy)
	print_evaluate(y, y_pred)

def testSdg(x,y):	
	# load the model from disk
	sgd_loaded_model = pickle.load(open('sgd_finalized_model.sav', 'rb'))
	y_pred= sgd_loaded_model.predict(x)
	rmse = np.sqrt(metrics.mean_squared_error(y, y_pred))
	accuracy=accuracy_score(y, y_pred)
	#print(rmse)
	#print(accuracy)
	print_evaluate(y, y_pred)

def testKmeans(x,y):	
	# load the cluster from disk
	kmeans_loaded_cluster = pickle.load(open('kmeans_finalized_cluster.sav', 'rb'))
	y_pred= kmeans_loaded_cluster.predict(x)
	rmse = np.sqrt(metrics.mean_squared_error(y, y_pred))
	accuracy=accuracy_score(y, y_pred)
	#print(accuracy)
	#print(rmse)
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

