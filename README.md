Machine-Learning-with-Spark-Streaming 
**Dataset** : **Sentiment Analysis**
---
Step 1. Run this command. It creates the base models while tuning them. These are saved as .sav files.
```Javascript
 python3 model.py
```
---
Training:
---
Step 2. Run stream.py using this command. It streams the .csv file in batches. It takes the name of dataset folder and the batch size as command line arguments. This opens a TCP connection and waits.

```Javascript
python3 stream.py -f sentiment -b 10000 
```
Step 3. Run stream_preprocess.py in a new terminal using this command. This file preprocesses the data being streamed from step 2, converts it into dataframes and RDDs, and puts them through the models defined in classification.py.
```Javascript
$SPARK_HOME/bin/spark-submit stream_preprocess.py > output.txt
```
---
Testing:
---
Step 4. Under the streamDataset function in stream.py, flip the comment status of the values in the DATASETS list. This ensures that the file now being streamed is test.csv.
Step 5. Run stream.py again using the command
```Javascript
python3 stream.py -f sentiment -b 10000 
```
Step 6. Run test.py using command. This file preprocesses the data being streamed from step 5, converts it into dataframes and RDDs, and puts them through the models defined in testing.py. It also displays accuracy, f1 score, and confusion matrix.

```Javascript
$SPARK_HOME/bin/spark-submit test.py > output_testing.txt
```
---
