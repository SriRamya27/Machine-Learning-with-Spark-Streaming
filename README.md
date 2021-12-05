##Machine-Learning-with-Spark-Streaming 
**dataset** :- **sentiment analysis**
---
1.**step 1**. run 
```Javascript
 python3 model.py
```
---
which will create all models we are working on and store it in form of .sav
2.**step 2**. run 
```python3 classfication.py ```
which has function which calls pickled model(stored model) and fits the data into the model 
---
3.**step3**. run stream.py using command
```python3 stream.py -f sentiment -b 10000 ```
which will send data in csv in batches it takes  name of folder where dataset is present and batch size as 4.command line arguements
---
**step4**.run stream_preprocess.py using command 
```$SPARK_HOME/bin/spark-submit stream_preprocess.py > output.txt```
which will collect data from stream.py through tcp connection converts it to rdd and does preprocess on data and call models from classification file. 
---
**step4**.run stream_preprocess.py using command 
```$SPARK_HOME/bin/spark-submit stream_preprocess.py > output.txt```
which will collect data from stream.py through tcp connection converts it to rdd and does preprocess on data and call models from classification file. 
**step5**.run sp1.py using command 
```$SPARK_HOME/bin/spark-submit stream_preprocess.py > output.txt```
which will collect data from stream.py through tcp connection converts it to rdd and does preprocess on data and call models from classification file. 
