###Machine-Learning-with-Spark-Streaming 
**dataset** :- **sentiment analysis**
---
1.**step 1**. run 
```Javascript
 python3 model.py
```
---

#which will create all models we are working on and store it in form of .sav
---
classfication.py 
#which has function which calls pickled model(stored model) and fits the data into the model 
---
3.**step2**. run stream.py using command
```Javascript
python3 stream.py -f sentiment -b 10000 
```
---
#which will send data in csv in batches it takes  name of folder where dataset is present and batch size as 4.command line arguements
---
4.**step3**.run stream_preprocess.py using command 
```Javascript
$SPARK_HOME/bin/spark-submit stream_preprocess.py > output.txt
```
#which will collect data from stream.py through tcp connection converts it to rdd and does preprocess on data and call models from classification file. 
---

##TESTING 
---
5.**step4**.comment line number *143* in  stream.py
---
6.**step5**.run stream.py again using command
```Javascript
python3 stream.py -f sentiment -b 10000 
```
---
7.**step6**.run test.py using command 
```Javascript
$SPARK_HOME/bin/spark-submit test.py > output_testing.txt
```
---
which will collect data from stream.py through tcp connection converts it to rdd and does preprocess on data and call models from classification file for prediction and displays accuracy, f1 score,confusion matrix
---
###PLOTS
---
```Javascript
python3 plot_bd.py
```
all experiment data like accuracy,rmse  is stored in folder called values  changes to be made is path to be made is changing the absolute path variable to absolute path to values folder .
