
from typing import Pattern
from pyspark.sql.types import *
from pyspark.sql.functions import *
# from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import Tokenizer, StopWordsRemover , IDF
from pyspark.sql import SparkSession, dataframe
import string
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer , RegexTokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline


#creating Spark session
appName = "Sentiment Analysis in Spark"
spark = SparkSession \
    .builder \
    .appName(appName) \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
#importing the csv directly for now (will incorporate the streaming element soon. This was done to get the preprocessing of data 
# out of the way early so the other modules can be focused on.)

#train.csv should be stored in the same directory as this file (TEMPORARY AS STREAMING WILL BE INCORPORATED)
tweets_csv = spark.read.csv('sentiment/train.csv', inferSchema=True, header=True)

# data = tweets_csv.select("Tweet", col("Sentiment").cast("Int").alias("Sentiment"))

#extracting words from the tweet in each row
tokenizer = RegexTokenizer(inputCol="Tweet", outputCol="SentimentWords" ,  pattern= '\\W')

tokenized = tokenizer.transform(tweets_csv)
#removing all useless words from the tokenised tweets and adding them to a list
add_stopwords = ["http" , "https" , "amp" , "rt" , "t" , "c" , "the" , "@" , "," , "-" , "com" , "an" , "of" , "for" , "ing" , "ed" , "tion"] #you can add any words to this list if you want it to be filtered out from the tweets
alphabet_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', \
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A'\
        , 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',\
             'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
add_stopwords = add_stopwords + alphabet_list
stopwordsRemover = StopWordsRemover(inputCol="SentimentWords", outputCol="filtered" ).setStopWords(add_stopwords)
# countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=5)
SwRemoved = stopwordsRemover.transform(tokenized)

# SwRemoved.show(truncate=False)

# SwRemoved.show(truncate=False) #uncomment this line if you want to see all columns, and then comment line 34 for easier viewing

#showing the final table only with columns "Tweet" , "Sentiment" , "MeaningfulWords"
final_table = SwRemoved.select("SentimentWords" , col("Sentiment").cast("Int").alias("Sentiment") , col("filtered").cast("string").alias("filtered"))

final_table.show(truncate=False) #refer line 30 if you want to see all the columns.


