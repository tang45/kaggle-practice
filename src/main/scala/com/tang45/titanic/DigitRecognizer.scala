package com.tang45.titanic

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SaveMode

object DigitRecognizer {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder
      .master("local[*]")
      .appName("digit")
      .getOrCreate()
    import spark.implicits._

    val sc = spark.sparkContext

    val data1 = sc.textFile("data/digit/train.csv")
    val data2 = sc.textFile("data/digit/test.csv")
    val fixData1 = dropHeader(data1)
    val fixData2 = dropHeader(data2)

    val trainData = fixData1
      .map { line =>
        val parts = line.split(",")
        val label = parts.head.toDouble
        val features = parts.tail.map(_.toDouble)
        LabeledPoint(label, Vectors.dense(features))
      }
      .cache()

    val lr = new LogisticRegressionWithLBFGS()
    lr.setIntercept(true).setNumClasses(10)
    lr.optimizer
      .setNumIterations(1000)
      .setConvergenceTol(0.001)
      .setRegParam(0.001)

    val model = lr.run(trainData)
    val testData = fixData2.map { line =>
      Vectors.dense(line.split(",").map(_.toDouble))
    }

    // predict result
    val res = model.predict(testData)
    val colName = Seq("ImageId", "Label")
    val df = res
      .zipWithIndex()
      .map { case (v,idx) =>
        (idx+1, v.toInt)
      }
      .toDF(colName: _*)

    // save csv file
    df.repartition(1)
      .write
      .format("csv")
      .option("header", "true")
      .mode(SaveMode.Append)
      .save("data/digit/")
  }

  def dropHeader(data: RDD[String]): RDD[String] = {
    val header = data.first()
    data.filter(_ != header)
  }
}
