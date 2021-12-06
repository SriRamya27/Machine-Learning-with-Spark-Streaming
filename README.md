Machine-Learning-with-Spark-Streaming 
**Dataset** : **Sentiment Analysis**
---
Step 1. Run this command.  
```Javascript
 python3 model.py
```
It creates the base models while tuning them. These are saved as .sav files.
---
Training:
---
Step 2. Run stream.py using this command.
```Javascript
python3 stream.py -f sentiment -b 10000 
```
This streams the .csv file in batches. It takes the name of dataset folder and the batch size as command line arguments. This opens a TCP connection and waits.
---
---
Step 3. Run stream_preprocess.py in a new terminal using this command. 
```Javascript
$SPARK_HOME/bin/spark-submit stream_preprocess.py > output.txt
```
This file preprocesses the data being streamed from step 2, converts it into dataframes, and puts them through the models defined in classification.py.
---
Testing:
---
Step 4. Under the streamDataset function in stream.py, flip the comment status of the values in the DATASETS list. This ensures that the file now being streamed is test.csv.
Step 5. Run stream.py again using the command
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
