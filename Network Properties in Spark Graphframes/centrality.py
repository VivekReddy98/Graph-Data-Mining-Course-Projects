from pyspark import SparkContext, Row
from pyspark.sql import SQLContext
from pyspark.sql import functions
from graphframes import *
from pyspark.sql.functions import explode, asc, desc
from itertools import combinations 
from pyspark.sql.types import FloatType
FloatType()

sc=SparkContext("local", "degree.py")
sqlContext = SQLContext(sc)


def closeness(g):
	# Get list of vertices. We'll generate all the shortest paths at
	# once using this list.
	vertices = g.vertices.rdd.collect()
	list_vertices = [i[0] for i in vertices]
	df = g.shortestPaths(landmarks=list_vertices)
	
	# Break up the map and group by ID for summing
	df = df.select(df["id"], explode(df["distances"]))
	df = df.groupBy("id").agg({'value' : 'sum'})

	# df.show()
	# Get the inverses and generate desired dataframe.
	df = df.select(df["id"], 1.0/df['sum(value)'])
	df = df.withColumnRenamed("(1.0 / sum(value))", "closeness")
	# df.show()
	return df


print("Reading in graph for problem 2.")
graph = sc.parallelize([('A','B'),('A','C'),('A','D'),('B','A'),('B','C'),('B','D'),('B','E'), ('C','A'),('C','B'),
	('C','D'),('C','F'),('C','H'),
	('D','A'),('D','B'),('D','C'),('D','E'),('D','F'),('D','G'),
	('E','B'),('E','D'),('E','F'),('E','G'),
	('F','C'),('F','D'),('F','E'),('F','G'),('F','H'),
	('G','D'),('G','E'),('G','F'),
	('H','C'),('H','F'),('H','I'),
	('I','H'),('I','J'),
	('J','I')])

e = sqlContext.createDataFrame(graph,['src','dst'])
v = e.selectExpr('src as id').union(e.selectExpr('dst as id')).distinct()
print("Generating GraphFrame.")
g = GraphFrame(v,e)

print("Calculating closeness.")
distrib = closeness(g)
# distrib.sort(desc("closeness")).show()

out = "centrality_out"
print("Writing distribution to file " + out + ".csv")
distrib.sort(desc("closeness")).toPandas().to_csv(out + ".csv")
