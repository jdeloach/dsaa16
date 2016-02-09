package org.arguslab.amandroid.ml

import scala.collection.mutable.ListBuffer
import org.json4s._
import org.json.JSONObject
import org.json.JSONArray
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.linalg.Vectors
import scala.reflect.io.File
import org.apache.spark.mllib.classification.NaiveBayesModel
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.classification.ClassificationModel
import org.apache.spark.ml.classification.ProbabilisticClassificationModel

object RandoomForest {
  def main(args : Array[String]) {
    val conf = new SparkConf()
      .setAppName("Random Forest")
      //.setMaster("spark://129.130.10.134:7077")
      .setMaster("local[4]")
    val sc = new SparkContext(conf)  

    // Load and parse the data file.
    val data = MLUtils.loadLibSVMFile(sc, "18to477.libsvm")
    // Split the data into training and test sets (30% held out for testing)
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))
    
    // Train a RandomForest model.
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 5 // Use more in practice.
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "gini"
    val maxDepth = 4
    val maxBins = 32
    
    val models = Range(1,8).map { x => RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth*x, maxBins) }

    models.map { model => {
      // Evaluate model on test instances and compute test error
      val labelAndPreds = testData.map { point =>
        val prediction = model.predict(point.features)
        (point.label, prediction)
      }
      
      val metrics = new BinaryClassificationMetrics(labelAndPreds)
      (model.trees(0).depth,metrics.areaUnderPR())
    }}.foreach{case (a,b) => println ("# Trees: " + a + ", auPRC: " + b)}
    
    //val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
    //println("Test Error = " + testErr)
    //println("Learned classification forest model:\n" + model.toDebugString)
    
    // Save and load model
    //model.save(sc, "target/tmp/myRandomForestClassificationModel")
    //val sameModel = RandomForestModel.load(sc, "target/tmp/myRandomForestClassificationModel")
  }
}