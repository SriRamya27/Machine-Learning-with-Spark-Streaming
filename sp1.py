#run this code using $SPARK_HOME/bin/spark-submit streaming_test.py. 
# for some weird reason the streaming doesn't work properly using python3, so this will do.

import numpy as np
import classification
#from sklearn.naive_bayes import BernoulliNB
#from sklearn.linear_model import Perceptron
#from sklearn.linear_model import SGDClassifier
#from sklearn.cluster import KMeans

from pyspark import SparkContext
import pyspark
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import RegexTokenizer , StopWordsRemover
import pyspark.sql.types as tp
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from sklearn.feature_extraction.text import HashingVectorizer
import re
import json
import testing

#removing all useless words and punctuations from the tokenised tweets 
add_stopwords = ["http" , "https" , "amp" , "rt" , "t" , "c" , "the" , "@" , "," , \
                "-" , "com" , "an" , "of" , "for" , "ing" , "ed" , "tion" , "&" , "quot"] #you can add any words to this list if you want it to be filtered out from the tweets
alphabet_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',  'm',\
                'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A'\
                , 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',\
                'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

add_stopwords = add_stopwords + alphabet_list


def display(rdd):

    sent = rdd.collect()
    if len(sent) > 0:
        try:
            df = spark.createDataFrame(json.loads(sent[0]).values() , schema = ["Sentiment" , "Tweet"])
            tokenizer = RegexTokenizer(inputCol="Tweet", outputCol="SentimentWords" ,  pattern= '\\W')
            stopwordsRemover = StopWordsRemover(inputCol="SentimentWords", outputCol="filtered" ).setStopWords(add_stopwords)
 
            x=df.select('Tweet').collect()
            x=[i['Tweet'] for i in x]
            vectorizer = HashingVectorizer(n_features=100000,stop_words=add_stopwords)
            x = vectorizer.fit_transform(x)
            #print(x)
            label_stringIdx = StringIndexer(inputCol="Sentiment" , outputCol="label")
            lix = label_stringIdx.fit(df.select("Sentiment"))
            lx = lix.transform(df.select("Sentiment"))
            y=lx.select('label').collect()
            y=np.array([i[0] for i in np.array(y)])
            #print(y)

            testing.testNaiveBayes(x,y)
            testing.testPerceptron(x,y)
            testing.testSdg(x,y)
            testing.testKmeans(x,y)
            
        except Exception as e:
        	print(e)
        	pass
        

if __name__ == "__main__":
    sc = SparkContext("local[2]", "PLEASEWORK")
    spark = SparkSession.builder.getOrCreate()
    ssc = StreamingContext(sc, 5)
    sql_context=SQLContext(sc)
    tweets = ssc.socketTextStream("localhost" , 6100)   

    words = tweets.flatMap(lambda line : line.split('\n'))
    # words = tweets.flatMap(lambda line : re.sub(r"http\S+" , "" , line).split('\n'))
    words.foreachRDD(display)


    ssc.start()
    ssc.awaitTermination()

    #you'll have to ctrl+Z or ctrl+C to stop this code from running (it doesn't end automatically after streaming is done)