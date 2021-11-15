# from pyspark import SparkContext
# from pyspark.streaming import StreamingContext

# sc = SparkContext(appName="TESTPLEASEWORK")
# ssc = StreamingContext(sc, 2)

# lines = ssc.socketTextStream("localhost" , 6100)


# lines.pprint()

# ssc.start()
# ssc.awaitTermination()
"""
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
import sys"""
#sc = SparkContext(master, appName)
# spark = SparkContext(appName="TESTPLEASEWORK")comment out for now
"""
spark = SparkContext.getOrCreate()

ssc = StreamingContext(spark,10000)"""
#lines = spark.readStream.format("socket").option("host","localhost").option("port", 6100).load()
#sqlContext = SQLContext(spark)
"""
lines = ssc.socketTextStream("localhost",6100)
print("lines type is " , type(lines))
words = lines.flatMap(lambda line: line.split(","))
words.pprint()
pairs = words.map(lambda word: (word, 1))
pairs.pprint()
wordCounts = pairs.reduceByKey(lambda x, y: x + y)
wordCounts.pprint()

print("computation starting now")

ssc.start() # Start the computation
ssc.awaitTermination() """

from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
import sys
#sc = SparkContext(master, appName)
spark = SparkContext.getOrCreate()
ssc = StreamingContext(spark,2)
#lines = spark.readStream.format("socket").option("host","localhost").option("port", 6100).load()
#sqlContext = SQLContext(spark)
lines = ssc.socketTextStream("localhost",6100)
words = lines.flatMap(lambda line: line.split(","))
pairs = words.map(lambda word: (word, 1))
wordCounts = pairs.reduceByKey(lambda x, y: x + y)
wordCounts.pprint()
ssc.start() # Start the computation
ssc.awaitTermination() 