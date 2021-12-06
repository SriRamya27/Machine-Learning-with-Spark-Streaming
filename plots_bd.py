import numpy as np
absolute_path='/home/pes1ug19cs124/Machine-Learning-with-Spark-Streaming-main'

kmean_accu_train= np.loadtxt(absolute_path+'/values/kmean_accu_train.txt')
kmean_accu_test= np.loadtxt(absolute_path+'/values/kmeans_accu_test.txt')
nb_accu_train= np.loadtxt(absolute_path+'/values/nb_train_accuracy.txt')
nb_accu_test= np.loadtxt(absolute_path+'/values/nb_test_accuracy.txt')
prep_accu_train= np.loadtxt(absolute_path+'/values/prep_acc_train.txt')
prep_accu_test= np.loadtxt(absolute_path+'/values/prep_acc_test.txt')
sgd_accu_train=np.loadtxt(absolute_path+'/values/sgd_train_accuracy.txt')
sgd_accu_test= np.loadtxt(absolute_path+'/values/sgd_test_accuracy.txt')

kmean_rmse_train= np.loadtxt(absolute_path+'/values/kmean_rmse_train.txt')
kmean_rmse_test= np.loadtxt(absolute_path+'/values/kmeans_rmse_test.txt')
nb_rmse_train= np.loadtxt(absolute_path+'/values/nb_train_rmse.txt')
nb_rmse_test= np.loadtxt(absolute_path+'/values/nb_test_rmse.txt')
sgd_rmse_train= np.loadtxt(absolute_path+'/values/sgd_train_rmse.txt')
sgd_rmse_test= np.loadtxt(absolute_path+'/values/sgd_test_rmse.txt')
prep_rmse_train= np.loadtxt(absolute_path+'/values/prep_rmse_train.txt')
prep_rmse_test= np.loadtxt(absolute_path+'/values/prep_rmse_test.txt')

kmean_accu_train_hyp= np.loadtxt(absolute_path+'/values/knn_accu_after_hyper.txt')

nb_accu_train_hyp= np.loadtxt(absolute_path+'/values/nb_train_hyperparameter.txt')

sgd_accu_train_hyp= np.loadtxt(absolute_path+'/values/sgd_train_hyperparameter.txt')

prep_accu_train_hyp=np.loadtxt(absolute_path+'/values/prep_accu_after_hyper.txt')

nb_1000=np.loadtxt(absolute_path+'/values/nb_train_hyperparameter_1000.txt')
nb_5000=np.loadtxt(absolute_path+'/values/nb_train_hyperparameter_5000.txt')
nb_10000=np.loadtxt(absolute_path+'/values/nb_train_hyperparameter.txt')
nb_50000=np.loadtxt(absolute_path+'/values/nb_train_hyperparameter_50000.txt')
nb_100000=np.loadtxt(absolute_path+'/values/nb_train_hyperparameter_100000.txt')
sgd_10000=np.loadtxt(absolute_path+'/values/sgd_train_accuracy.txt')
sgd_50000=np.loadtxt(absolute_path+'/values/sgd_train_50000.txt')
sgd_100000=np.loadtxt(absolute_path+'/values/sgd_train_100000.txt')

"""##KMEAN MINI BATCHES

1.PLOT BETWEEN ACCURACY AND BATCHES FOR BATCH SIZE =10000
"""

import matplotlib.pyplot as plt
def kmeans_batch_vs_accuracy():
	plt.title("KMEANS")
	plt.xlabel("BATCHES")
	plt.ylabel("ACCURACY")
	plt.grid()
	plt.plot(kmean_accu_train,color='green',marker = 'o',label='training data')
	plt.plot(kmean_accu_test,color='red',marker = '*',label='testing data')
	plt.show()

"""2.PLOT BETWEEN ROOT MEAN SQUARED ERROR  AND BATCHES FOR BATCH SIZE =10000"""

def kmeans_batch_vs_rmse():
	plt.title("KMEANS")
	plt.xlabel("BATCHES")
	plt.ylabel("Root mean squared error")
	plt.grid()
	plt.plot(kmean_rmse_train,color='green',marker = 'o',label='training data')
	plt.plot(kmean_rmse_test,color='red',marker = '.',label='testing data')
	plt.show()

def kmeans_hyperparameter():
	plt.title("kmeans with hyperparameter tuning")
	plt.xlabel("BATCHES")
	plt.ylabel("ACCURACY")
	plt.grid()
	plt.plot(kmean_accu_train,color='green',marker = 'o',label='training data')
	plt.plot(kmean_accu_train_hyp,color='red',marker = '.',label='testing')
	plt.show()

"""##NAIVE BAYES

1.PLOT BETWEEN ACCURACY AND BATCHES FOR BATCH SIZE =10000
"""

def nb_batch_vs_accuracy():
	plt.title("Naive Bayes")
	plt.xlabel("BATCHES")
	plt.ylabel("ACCURACY")
	plt.grid()
	plt.plot(nb_accu_train,color='green',marker = 'o',label='training data')
	plt.plot(nb_accu_test,color='red',marker = '.',label='testing data')
	plt.show()

def nb_batch_vs_rmse():
	plt.title("Naive Bayes")
	plt.xlabel("BATCHES")
	plt.ylabel("root mean squared error")
	plt.grid()
	plt.plot(nb_rmse_train,color='green',marker = 'o',label='training data')
	plt.plot(nb_rmse_test,color='red',marker = '.',label='testing data')
	plt.show()

def nb_hyper():
	plt.title("Naive Bayes with hyper parameter tuning")
	plt.xlabel("BATCHES")
	plt.ylabel('Accuracy')
	plt.grid()
	plt.plot(nb_accu_train,color='green',marker = 'o',label='training data')
	plt.plot(nb_accu_train_hyp,color='red',marker = '.',label='with hyperparameter')
	plt.show()
#nb_batch_vs_accuracy()
#nb_batch_vs_rmse()
#nb_hyper()
"""##PERCEPTRON"""

def perceptron_batch_vs_accuracy():
	plt.title("PERCEPTRON")
	plt.xlabel("BATCHES")
	plt.ylabel("ACCURACY")
	plt.grid()
	plt.plot(prep_accu_train,color='green',marker = 'o',label='training data')
	plt.plot(prep_accu_test,color='red',marker = '.',label='testing data')
	plt.show()

def perceptron_batch_vs_rmse():
	plt.title("PERCEPTRON")
	plt.xlabel("BATCHES")
	plt.ylabel("RMSE")
	plt.grid()
	plt.plot(prep_rmse_train,color='green',marker = 'o',label='training data')
	plt.plot(prep_rmse_test,color='red',marker = '.',label='testing data')
	plt.show()
def preceptron_hyper():
	plt.title("PERCEPTRON WITH HYPERPARAMETER TUNING")
	plt.xlabel("BATCHES")
	plt.ylabel("ACCURACY")
	plt.grid()
	plt.plot(prep_accu_train,color='green',marker = 'o',label='training data')
	plt.plot(prep_accu_train_hyp,color='red',marker = '.',label='with hyperparameter')
	plt.show()
#perceptron_batch_vs_accuracy()
#perceptron_batch_vs_rmse()
#preceptron_hyper()

"""##SGD"""
def sgd_batch_vs_accuracy():
	plt.title("SGD")
	plt.xlabel("BATCHES")
	plt.ylabel("ACCURACY")
	plt.grid()
	plt.plot(sgd_accu_train,color='green',marker = 'o',label='training data')
	plt.plot(sgd_accu_test,color='red',marker = '.',label='testing')
	plt.show()

def sgd_batch_vs_rmse():
	plt.title("SGD")
	plt.xlabel("BATCHES")
	plt.ylabel("ROOT MEAN SQUARED ERROR")
	plt.grid()
	plt.plot(sgd_rmse_train,color='green',marker = 'o',label='training data')
	plt.plot(sgd_rmse_test,color='red',marker = '.',label='testing')
	plt.show()

def sgd_hyper():
	plt.title("SGD with hyperparameter tuning")
	plt.xlabel("BATCHES")
	plt.ylabel("ACCURACY")
	plt.grid()
	plt.plot(sgd_accu_train,color='green',marker = 'o',label='training data')
	plt.plot(sgd_accu_train_hyp,color='red',marker = '.',label='testing')
	plt.show()
#sgd_hyper()
#sgd_batch_vs_rmse()
#sgd_batch_vs_accuracy()
"""##BATCH SIZE EXPERIMENT WITH NAIVE BAYES(non linear)"""

def batch_decreasing_nonlinear():
	plt.title("BATCH SIZE EXPERIMENT")
	plt.xlabel("BATCHES")
	plt.ylabel("ACCURACY")
	plt.grid()
	plt.plot(nb_10000,color='green',marker = 'o',label='1000')
	plt.plot(nb_5000,color='blue',marker = 'o',label='5000')
	plt.plot(nb_1000,color='red',marker = '.',label='10000')
	plt.show()

def batch_increasing_nonlinear():
	plt.title("BATCH SIZE EXPERIMENT")
	plt.xlabel("BATCHES")
	plt.ylabel("ACCURACY")
	plt.grid()
	plt.plot(nb_100000,color='green',marker = 'o',label='1000')
	plt.plot(nb_50000,color='blue',marker = 'o',label='5000')
	plt.plot(nb_10000,color='red',marker = '.',label='10000')
	plt.show()

def batch_increasing_linear():
	plt.title("BATCH SIZE EXPERIMENT")
	plt.xlabel("BATCHES")
	plt.ylabel("ACCURACY")
	plt.grid()
	plt.plot(sgd_100000,color='green',marker = 'o',label='1000')
	plt.plot(sgd_50000,color='blue',marker = 'o',label='5000')
	plt.plot(sgd_10000,color='red',marker = '.',label='10000')
	plt.show()

#batch_increasing_linear()
#batch_increasing_nonlinear()
#batch_decreasing_nonlinear()
