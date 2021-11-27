#run this code using $SPARK_HOME/bin/spark-submit streaming_test.py. 
# for some weird reason the streaming doesn't work properly using python3, so this will do.

from pyspark import SparkContext
import pyspark
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.feature import RegexTokenizer , StopWordsRemover
import pyspark.sql.types as tp
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline, pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.functions import * 

import re
import json

#removing all useless words and punctuations from the tokenised tweets 
add_stopwords = ["rt" , "the" , "an" , "of" , "for" , "that" , "is" , "was" \
                 , "will", "has" , "have" , "had" , "and", "with" ,
                 "can", "it" , "so" , "am" , "be" ,"to", "wasn" , "," , "-", "." ] #you can add any words to this list if you want it to be filtered out from the tweets
alphabet_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',  'm',\
                'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A'\
                , 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',\
                'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

add_stopwords = add_stopwords + alphabet_list


def display(rdd):

    sent = rdd.collect()
    # print(sent)
    # sent = remove_url(sent)
    if len(sent) > 0:
        try:
            df = spark.createDataFrame(json.loads(sent[0]).values() , schema = ["Sentiment" , "Tweet"])
            df = df.withColumn("Tweet" , regexp_replace("Tweet" , r"http\S+", ""))
            df = df.withColumn("Tweet" , regexp_replace("Tweet" , r"@\S+", ""))
            tokenizer = RegexTokenizer(inputCol="Tweet", outputCol="SentimentWords" ,  pattern= '\\W')
            stopwordsRemover = StopWordsRemover(inputCol="SentimentWords", outputCol="filtered" ).setStopWords(add_stopwords)
            # stopwordsRemover2 = StopWordsRemover(inputCol="filtered", outputCol="double_filtered" ).setStopWords(add_stopwords)
            hashtf = HashingTF(numFeatures=2**16 , inputCol="filtered" , outputCol="tf")
            idf = IDF(inputCol="tf" , outputCol="features" , minDocFreq= 5)
            label_stringIdx = StringIndexer(inputCol="Sentiment" , outputCol="label") #by default, 0 (from dataset) is mapped to 1, and 4 to 0
            pipeline = Pipeline(stages=[tokenizer , stopwordsRemover , hashtf , idf , label_stringIdx])
            pipelineFit = pipeline.fit(df)
            train_df = pipelineFit.transform(df)
            train_df.show(truncate=False , n = 5)
        except:
            print("mistake")
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

    
