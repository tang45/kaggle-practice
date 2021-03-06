package com.tang45.titanic

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SaveMode
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.DataFrame

object SurviveStat {
  def main(args: Array[String]) {

    val sparkSession = SparkSession.builder
      .master("local[*]")
      .appName("titanic")
      .getOrCreate()

    sparkSession.sparkContext.setLogLevel("ERROR")

    //load train df
    val df = sparkSession.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("data/titanic/train.csv")

    //handle missing values
    val meanValue = df.agg(mean(df("Age"))).first.getDouble(0)
    val fixedDf = df.na.fill(meanValue, Array("Age"))

    //extract and transfer values
    val extractDf =
      fixedDf.withColumn("Name",
                         regexp_extract(col("Name"), "(, )([^\\.]+)", 2))
    val transformName = udf { (name: String) =>
      if (name == "Jonkheer")
        "Master"
      else if (name == "Ms")
        "Miss"
      else if (name == "Mlle")
        "Miss"
      else if (name == "Mme")
        "Mrs"
      else if (name == "Dona")
        "Mrs"
      else if (name == "Lady")
        "Mrs"
      else if (name == "the Countess")
        "Mrs"
      else if (name == "Capt")
        "Mr"
      else if (name == "Don")
        "Mr"
      else if (name == "Major")
        "Mr"
      else if (name == "Col")
        "Mr"
      else if (name == "Sir")
        "Mr"
      else if (name == "Capt")
        "DrAndRev"
      else if (name == "Rev")
        "DrAndRev"
      else name
    }
    val transferDf =
      extractDf.withColumn("Name", transformName(col("Name")))

    //test and train split
    val dfs = transferDf.randomSplit(Array(0.7, 0.3))
    val trainDf = dfs(0).withColumnRenamed("Survived", "label")
    val crossDf = dfs(1)

    // create pipeline stages for handling categorical
    val genderStages = handleCategorical("Sex")
    val embarkedStages = handleCategorical("Embarked")
    val pClassStages = handleCategorical("Pclass")
    val nameStages = handleCategorical("Name")

    //columns for training
    val cols = Array("Sex_onehot",
                     "Embarked_onehot",
                     "Pclass_onehot",
                     "Name_onehot",
                     "SibSp",
                     "Parch",
                     "Age",
                     "Fare")
    val vectorAssembler =
      new VectorAssembler().setInputCols(cols).setOutputCol("features")

    //algorithm stage
    val randomForestClassifier = new RandomForestClassifier()
    //pipeline
    val preProcessStages = nameStages ++ genderStages ++ embarkedStages ++ pClassStages ++ Array(vectorAssembler)
    val pipeline = new Pipeline()
      .setStages(preProcessStages ++ Array(randomForestClassifier))

    val model = pipeline.fit(trainDf)
    println(
      "train accuracy with pipeline: " + accuracyScore(model.transform(trainDf),
                                                       "label",
                                                       "prediction"))
    println(
      "test accuracy with pipeline: " + accuracyScore(model.transform(crossDf),
                                                      "Survived",
                                                      "prediction"))

    //cross validation
    val paramMap = new ParamGridBuilder()
      .addGrid(randomForestClassifier.impurity, Array("gini", "entropy"))
      .addGrid(randomForestClassifier.maxDepth, Array(1, 2, 5, 10, 15))
      .addGrid(randomForestClassifier.minInstancesPerNode,
               Array(1, 2, 4, 5, 10))
      .build()

    val cvModel = crossValidation(pipeline, paramMap, trainDf)
    println(
      "train accuracy with cross validation: " + accuracyScore(
        cvModel.transform(trainDf),
        "label",
        "prediction"))
    println(
      "test accuracy with cross validation: " + accuracyScore(
        cvModel.transform(crossDf),
        "Survived",
        "prediction"))

    val testDf = sparkSession.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("data/titanic/test.csv")
    val extractTestDf =
      testDf.withColumn("Name",
        regexp_extract(col("Name"), "(, )([^\\.]+)", 2))
    val transferTestDf =
      extractTestDf.withColumn("Name", transformName(col("Name")))
    val fareMeanValue = df.agg(mean(df("Fare"))).first.getDouble(0)
    val fixedOutputDf = transferTestDf.na
      .fill(meanValue, Array("age"))
      .na
      .fill(fareMeanValue, Array("Fare"))

    generateOutputFile(fixedOutputDf, model)
  }

  def generateOutputFile(testDF: DataFrame, model: Model[_]) = {
    val scoredDf = model.transform(testDF)
    val outputDf = scoredDf.select("PassengerId", "prediction")
    val castedDf = outputDf.select(
      outputDf("PassengerId"),
      outputDf("prediction").cast(IntegerType).as("Survived"))
    castedDf.write
      .format("csv")
      .option("header", "true")
      .mode(SaveMode.Append)
      .save("data/titanic/")
  }

  def crossValidation(pipeline: Pipeline,
                      paramMap: Array[ParamMap],
                      df: DataFrame): Model[_] = {
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramMap)
      .setNumFolds(5)
    cv.fit(df)
  }

  def handleCategorical(column: String): Array[PipelineStage] = {
    val stringIndexer = new StringIndexer()
      .setInputCol(column)
      .setOutputCol(s"${column}_index")
      .setHandleInvalid("skip")
    val oneHot = new OneHotEncoder()
      .setInputCol(s"${column}_index")
      .setOutputCol(s"${column}_onehot")
    Array(stringIndexer, oneHot)
  }

  def accuracyScore(df: DataFrame, label: String, predictCol: String) = {
    val rdd = df
      .select(predictCol, label)
      .rdd
      .map(row ⇒ (row.getDouble(0), row.getInt(1).toDouble))
    new MulticlassMetrics(rdd).accuracy
  }
}
