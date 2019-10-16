import sys
import time
import networkx as nx
from pyspark import SparkContext, Row
from pyspark.sql import SQLContext
from pyspark.sql import functions
from pyspark.sql.functions import * 
from graphframes import *
from copy import deepcopy
import pandas as pd
from pyspark.streaming import StreamingContext


sc=SparkContext("local", "degree.py")
sqlContext = SQLContext(sc)
ssc = StreamingContext(sc, 6)
ssc.checkpoint("./checkpoint")

def articulations(g, usegraphframe=False):
	# Get the starting count of connected components
	G = nx.from_pandas_edgelist(g.edges.toPandas(), source="src", target="dst")
	# print([x for x in nx.articulation_points(G)])
	Total_cc = nx.number_connected_components(G)

	# Default version sparkifies the connected components process 
	# and serializes node iteration.
	if usegraphframe:
		# Get vertex list for serial iteration
		vertex_list = g.vertices.collect()
			
		# For each vertex, generate a new graphframe missing that vertex
		# and calculate connected component count. Then append count to
		# the output
		# YOUR CODE HERE
		Row_obj = []
		for i in nx.articulation_points(G):
			Rdd = g.edges.rdd.map(lambda l: Row(src=l['dst'], dst=l['dst']) if l['src']==i else l)
			Rdd = Rdd.map(lambda l: Row(src=l['src'], dst=l['src']) if l['dst']==i else l)
			Rdd = sqlContext.createDataFrame(Rdd)
			G_new = GraphFrame(g.vertices, Rdd)
			result = G_new.connectedComponents()
			articulation = 1 if result.select('component').distinct().count()>Total_cc else 0
			Row_obj.append(Row(id=i, articulation= articulation))
			print(i, result.select('component').distinct().count())
		df = sc.parallelize(Row_obj)
		df = sqlContext.createDataFrame(df)
		
		
	# Non-default version sparkifies node iteration and uses networkx 
	# for connected components count.
	else:
	 # YOUR CODE HERE
		def remove_Node(x, rdd):
			for i in range(0,rdd.shape[0]):
				if rdd['src'][i] == x:
					rdd['src'][i] = rdd['dst'][i]
				elif rdd['dst'][i] == x:
					rdd['dst'][i] = rdd['src'][i]
			return rdd
		df = pd.DataFrame(columns=['id', 'articulation'])
		vertices_iter = g.vertices.rdd.map(lambda x: x.id).toLocalIterator()
		
		for x in vertices_iter:
			rdd = g.edges.toPandas()
			rdd = remove_Node(x, rdd)
			G = nx.from_pandas_edgelist(rdd, source="src", target="dst")
			CC = 1 if nx.number_connected_components(G)>Total_cc else 0
			df = df.append({'id':x, 'articulation':CC}, ignore_index=True)
		df = sqlContext.createDataFrame(df)
	return df
       

filename = sys.argv[1]
lines = sc.textFile(filename)

pairs = lines.map(lambda s: s.split(","))
e = sqlContext.createDataFrame(pairs,['src','dst'])
e = e.unionAll(e.selectExpr('src as dst','dst as src')).distinct() # Ensure undirectedness 	

# Extract all endpoints from input file and make a single column frame.
v = e.selectExpr('src as id').unionAll(e.selectExpr('dst as id')).distinct()	


# Create graphframe from the vertices and edges.
g = GraphFrame(v,e)

print("Runtime approximately 5 minutes")
print("---------------------------")
print("Processing graph using Spark iteration over nodes and serial (networkx) connectedness calculations")
init = time.time()
df = articulations(g, False)
df.toPandas().to_csv("centrality.csv")
print("Execution time: %s seconds" % (time.time() - init))
print("Articulation points:")
df.filter('articulation = 1').show(truncate=False)

print("---------------------------")

#Runtime for below is more than 2 hours
print("Processing graph using serial iteration over nodes and GraphFrame connectedness calculations")
init = time.time()
df = articulations(g, True)
print("Execution time: %s seconds" % (time.time() - init))
print("Articulation points:")
df.filter('articulation = 1').show(truncate=False)
