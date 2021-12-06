Machine-Learning-with-Spark-Streaming 
**Dataset** : **Sentiment Analysis**
---
Step 1. Run this command.  
```Javascript
 python3 model.py
```
---
It creates the base models while tuning them. These are saved as .sav files.
---
Step 2. Run stream.py using command
```Javascript
python3 stream.py -f sentiment -b 10000 
```
---
#which will send data in csv in batches it takes  name of folder where dataset is present and batch size as 4.command line arguements
---
Step 4. Run stream_preprocess.py using command 
```Javascript
$SPARK_HOME/bin/spark-submit stream_preprocess.py > output.txt
```
#which will collect data from stream.py through tcp connection converts it to rdd and does preprocess on data and call models from classification file. 
---
##TESTING 
---
5.**step5**.comment line number *143* in  stream.py
---
6.**step6**.run stream.py again using command
```Javascript
python3 stream.py -f sentiment -b 10000 
```
---
7.**step7**.run sp1.py using command 
```Javascript
$SPARK_HOME/bin/spark-submit sp1.py > output_testing.txt
```
---
which will collect data from stream.py through tcp connection converts it to rdd and does preprocess on data and call models from classification file for prediction and displays accuracy, f1 score,confusion matrix
---
###PLOTS
...
---
