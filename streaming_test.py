#run this code using $SPARK_HOME/bin/spark-submit streaming_test.py. 
# for some weird reason the streaming doesn't work properly using python3, so this will do.

from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql import Row
import pyspark.sql.types as tp
import json


def display(rdd):

    sent = rdd.collect()
    if len(sent) > 0:
        df = spark.createDataFrame(json.loads(sent[0]).values() , schema = ["sentiment" , "tweet"])
        df.show(truncate=False)
	


if __name__ == "__main__":
    sc = SparkContext("local[2]", "PLEASEWORK")
    spark = SparkSession.builder.getOrCreate()
    ssc = StreamingContext(sc, 5)
    sql_context=SQLContext(sc)
    tweets = ssc.socketTextStream("localhost" , 6100)   

    words = tweets.flatMap(lambda line : line.lower().split('\n'))
    words.foreachRDD(display)


    ssc.start()
    ssc.awaitTermination()

    
