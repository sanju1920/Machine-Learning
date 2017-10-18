import org.apache.spark.ml.regression.LinearRegression
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

import org.apache.spark.sql.SparkSession

val spark=SparkSession.builder().getOrCreate()

val data =spark.read.option("header","true").option("inferSchema","true").format("csv").load("Clean-USA-Housing.csv")

data.printSchema

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors


val df =data.select(data("Price").as("label"),$"Avg Area Income",$"Avg Area House Age", $"Avg Area Number of Rooms", $"Avg Area Number of Bedrooms", $"Area Population")

val assembler = new VectorAssembler().setInputCols(Array("Avg Area Income","Avg Area House Age","Avg Area Number of Rooms","Avg Area Number of Bedrooms","Area Population")).setOutputCol("features")

val output = assembler.transform(df).select($"label",$"features")

/// ////////////////// UPTO THIS IS THIS IS SAME FOR ALL THE DATE WE JUST TAKE INPUT THE DATA FILE AND CONVERT THE FILE IN VECTOR INPUT AND OUTPUT SET////

// NOW TIME FOR ALGORITHM  /////

val lr = new LinearRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)

 val m = lr.fit(output)

 val training = m.summary

 training.residuals.show()

  println(s"RMSE: ${training.rootMeanSquaredError}")
   println(s"RM: ${training.meanSquaredError}")
    println(s"R2: ${training.r2}")