import sys
import pandas
import numpy as np
import networkx as nx
from pyspark import SparkContext, Row
from pyspark.sql import SQLContext
from pyspark.sql.functions import * 
from pyspark.sql.types import *
from graphframes import *
import pandas as pd

sc=SparkContext("local", "degree.py")
sqlContext = SQLContext(sc)


''' return the simple closure of the graph as a graphframe.'''
def simple(g):
	# Extract edges and make a data frame of "flipped" edges
	row_data = g.edges.rdd.map(lambda l: Row(src=l[0], dst=l[1]))
	flipped_Edges = sqlContext.createDataFrame(row_data)
	# print("flipped graph has " + str(flipped_Edges.count()))
	combined_edges = g.edges.union(flipped_Edges)  #<2.0, the command is unionAll
	# print("Multi Combined graph has " + str(combined_edges.count()))
	combined_edges = combined_edges.dropDuplicates()

	# Combine old and new edges. Distinctify to eliminate multi-edges
	# Filter to eliminate self-loops.
	# A multigraph with loops will be closured to a simple graph
	# If we try to undirect an undirected graph, no harm done
	# print("Unfiltered and combined graph has" + str(GraphFrame(g.vertices,combined_edges).edges.count()))
	return GraphFrame(g.vertices,combined_edges).filterEdges("src != dst")

def degreedist(g):
	# Generate a DF with degree,count
	return g.outDegrees.groupBy('outdegree').count().sort(asc("outdegree"))


def readFile(filename, large, sqlContext=sqlContext):
	lines = sc.textFile(filename)

	if large:
		delim=" "
		# Strip off header row.
		lines = lines.mapPartitionsWithIndex(lambda ind,it: iter(list(it)[1:]) if ind==0 else it)
	else:
		delim=","

	lines = lines.map(lambda l : l.strip().split(delim))
	row_data = lines.map(lambda l: Row(src=int(l[0]), dst=int(l[1])))
	Edges = sqlContext.createDataFrame(row_data)

	print("Edges Data Frame was Created\n")

	lines = lines.flatMap(lambda l: map(int, l))
	lines = lines.distinct()
	row_data = lines.map(lambda p: Row(id=int(p)))
	Vertices = sqlContext.createDataFrame(row_data)
	# Vertices.take(5).foreach(println)
	print("Vertices Dataframe was also created\n")
	
	# Create graphframe g from the vertices and edges.
	G = GraphFrame(Vertices,Edges)
	print("Graph Dataframe is Created")
	# G.vertices.show()
	return G

# main stuff

# If you got a file, yo, I'll parse it.
if len(sys.argv) > 1:
	filename = sys.argv[1]
	if len(sys.argv) > 2 and sys.argv[2]=='large':
		large=True
	else:
		large=False

	print("Processing input file " + filename)
	g = readFile(filename, large)

	print("Original graph has " + str(g.edges.count()) + " directed edges and " + str(g.vertices.count()) + " vertices.")
	
	g2 = simple(g)
	# print(g2.edges.show())
	print("Simple graph has " + str(g2.edges.count()/2) + " undirected edges.")

	distrib = degreedist(g2)
	distrib.show()
	nodecount = g2.vertices.count()
	print("Graph has " + str(nodecount) + " vertices.")

	out = filename.split("/")[-1]
	print("Writing distribution to file " + out + ".csv")
	distrib.toPandas().to_csv(out + ".csv")

# Otherwise, generate some random graphs.
else:
	print("Generating random graphs.")
	vschema = StructType([StructField("id", IntegerType())])
	eschema = StructType([StructField("src", IntegerType()),StructField("dst", IntegerType())])

	gnp1 = nx.gnp_random_graph(100, 0.05, seed=1234)
	gnp2 = nx.gnp_random_graph(2000, 0.01, seed=5130303)
	gnm1 = nx.gnm_random_graph(100,1000, seed=27695)
	gnm2 = nx.gnm_random_graph(1000,100000, seed=9999)

	todo = {"gnp1": gnp1, "gnp2": gnp2, "gnm1": gnm1, "gnm2": gnm2}
	for gx in todo:
		print("Processing graph " + gx)
		v_rdd = sc.parallelize(todo[gx].nodes())
		row_data = v_rdd.map(lambda l: Row(id=l))
		v = sqlContext.createDataFrame(row_data)
		e_rdd = sc.parallelize(todo[gx].edges())
		row_data = e_rdd.map(lambda l: Row(src=l[0], dst=l[1]))
		e = sqlContext.createDataFrame(row_data)
		g = simple(GraphFrame(v,e))
		print("Original graph has " + str(g.edges.count()))
		print(" directed edges and " + str(g.vertices.count()) + " vertices.")
		distrib = degreedist(g)
		print("Writing distribution to file " + gx + ".csv")
		distrib.toPandas().to_csv(gx + ".csv") 
