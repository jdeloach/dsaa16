package org.arguslab.amandroid.ml

import java.io.File
import java.io.FileOutputStream
import java.io.PrintWriter
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.classification.NaiveBayesModel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import scala.collection.mutable.LinkedList
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.classification.SVMWithSGD


/**
 * @author jdeloach
 */
object NaiveBayesCotraining {
  def main(args: Array[String]) {
    val conf = new SparkConf()
      .setAppName("Casual Naive Bayes EM")
      //.setMaster("spark://129.130.10.134:7077")
      .setMaster("local[3]")
    val sc = new SparkContext(conf)    
    
    /*val data = sc.textFile("/Users/jdeloach/Code/Weka/weka-3-7-12/18to477.libsvm")
    val parsedData = data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }
    */
    val partitions = 8
    val parsedData = MLUtils.loadLibSVMFile(sc, "/Users/jdeloach/Code/PyBrain/subset.libsvm", 471, partitions) ///Users/jdeloach/Code/Weka/weka-3-7-12/
    /** 
     * Changed partitions from 24, and file url, and .setMaster
     */
    
    // Split data into training (60%) and test (40%).
    val splits = parsedData.randomSplit(Array(0.66, 0.33), seed = 11L)
    val trainingBase = splits(0).cache
    var training = trainingBase.sample(false, .3, seed = 58L).cache // takes 10% of 60% or 6% as base training dataset
    println("Number of negative examples in training: " + training.collect().count(p => p.label == 1) + ", num pos:" + training.collect().count(_.label == 0))
    Thread.sleep(1000)
    var remainingTraining = trainingBase.subtract(training).cache//.collect()
    var test = splits(1).cache
    
    //val count = remainingTraining.size
    val count = remainingTraining.count
    
    for(i <- 1 to 10) { // iterations
      val model = NaiveBayes.train(training, lambda = 1.0)    
      val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
      val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()
      val metrics = new BinaryClassificationMetrics(predictionAndLabel)

      println("accuracy after iteration " + i + ": " + accuracy)  
      val out = new PrintWriter(new FileOutputStream(new File("spark_em_algo.out"), true))
      out.println("accuracy after iteration " + i + ": " + accuracy + ", auPRC:" + metrics.areaUnderPR())  

      val selectSize = (.025 * count).toInt
      // sort local
   /*   val mostPerPart0 = remainingTraining.mapPartitions(_.toList.map({ x => (x,model.predictProbabilities(x.features)(0)) }).sortBy(f => f._2).reverse.slice(0,selectSize).toIterator, false).collect()
      // sort global
      val mostConf0 = mostPerPart0.sortBy(_._2).reverse.slice(0, selectSize).map(_._1) 
      val mostPerPart1 = remainingTraining.mapPartitions(_.toList.map({ x => (x,model.predictProbabilities(x.features)(1)) }).sortBy(f => f._2).reverse.slice(0,selectSize).toIterator, false).collect()
      val mostConf1 = mostPerPart1.sortBy(_._2).reverse.slice(0, selectSize).map(_._1)
    */  
      //val mostConf0 = remainingTraining.top((.025 * count).toInt)(Ordering.by(p => model.predictProbabilities(p.features)(0)))
      //val mostConf1 = remainingTraining.top((.025 * count).toInt)(Ordering.by(p => model.predictProbabilities(p.features)(1)))
      
      //val mostConf0 = remainingTraining.sortBy(p => model.predictProbabilities(p.features)(0), false, 48).take((.05 * count).toInt)
      //val mostConf1 = remainingTraining.sortBy(p => model.predictProbabilities(p.features)(1), false, 48).take((.05 * count).toInt)
      //val mostConf0 = remainingTraining.map { p => (p,model.predictProbabilities(p.features)) }.sortBy(_._2(0), false).take((.05 * count).toInt).map(_._1)     
      //val mostConf1 = remainingTraining.map { p => (p,model.predictProbabilities(p.features)) }.sortBy(_._2(1), false).take((.05 * count).toInt).map(_._1)
      
    //  val mostConf0 = remainingTraining.sortBy { p => 1 - model.predictProbabilities(p.features)(0) }.take(selectSize) 
    //  val mostConf1 = remainingTraining.sortBy { p => 1 - model.predictProbabilities(p.features)(1) }.take(selectSize) 
      val mostConf = remainingTraining.sortBy { x => {
        val probs = model.predictProbabilities(x.features)
        Math.max(1 - probs(0), 1 - probs(1))
      }}.take(selectSize).distinct
      
   //   mostConf0.foreach { x => println("Our guess for 0 is actually a :" + x.label + ", prob:" + model.predictProbabilities(x.features)) }
   //   mostConf1.foreach { x => println("Our guess for 1 is actually a :" + x.label + ", prob:" + model.predictProbabilities(x.features)) }
      
      mostConf.foreach { x => out.println("Our guess is " + model.predictProbabilities(x.features) + ", actually is: " + x.label) }
      out.close
      
      // take the 5% we are most 
      val newTrain = sc.parallelize(mostConf)
      training = training.union(newTrain).cache()
      remainingTraining = trainingBase.subtract(newTrain).cache()
    } 
    
    // Save and load model
    //model.save(sc, "models")
    //val sameModel = NaiveBayesModel.load(sc, "models")
  }
}
