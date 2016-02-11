package org.arguslab.amandroid.ml

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import java.io.FileOutputStream
import java.io.PrintWriter
import java.io.File
import experiments.ensemble.NaiveBayesBinaryVoting

object EasyEnsemble {
  def main(args: Array[String]) : Unit = {
    val conf = new SparkConf()
    .setAppName("EasyEnsemble")
    .setMaster("local[3]")
    val sc = new SparkContext(conf)  
    
    val baseData = MLUtils.loadLibSVMFile(sc, "subset.libsvm", 471, 7).cache ///Users/jdeloach/Code/Weka/weka-3-7-12/
    val testData = MLUtils.loadLibSVMFile(sc, "18to477.libsvm").sample(false, .002, 11L).cache
    val negativeSet = baseData.filter { x => x.label == 1 && x.features.numNonzeros > 0 }.collect
    val positiveSet = baseData.filter { x => x.label == 0 }.cache
    val positiveCount = positiveSet.count.toInt
    val negativeCount = negativeSet.length.toInt
    val subset = negativeCount/baseData.count.toDouble
    
    val numClassifiers = (Math.floor(positiveCount / negativeCount)).toInt/2
    /*val positiveSets = positiveSet.collect().grouped(negativeCount).toList
    val trainings = positiveSets.map { x => sc.parallelize(x.union(negativeSet)) }*/
    val trainings = Range(0,numClassifiers).map { x => positiveSet.sample(false, subset, 11L).union(sc.parallelize(negativeSet)) }
    var remainingTrainings = positiveSet.subtract(trainings.reduce((a,b) => a.union(b))).collect
    
    val models = trainings.map { x => {
      var m = NaiveBayes.train(x)
      /*var localTrainings = x.collect()
      var localRemainingTrainings = remainingTrainings.clone()
      val totalExtraTrainings = localRemainingTrainings.length
      val iterations = 10
      
      for(i <- 1 to iterations) {
        val toTrain = localRemainingTrainings.sortBy { x => {
          val pred = m.predictProbabilities(x.features)
          Math.max(pred(0), pred(1))
        }}.reverse.take(totalExtraTrainings/iterations)
        localTrainings = localTrainings.union(toTrain)
        m = NaiveBayes.train(sc.parallelize(localTrainings))
        val raw = (new BinaryClassificationMetrics(testData.map(p => (p.label,m.predict(p.features)))).areaUnderPR())
        val out = new PrintWriter(new FileOutputStream(new File("spark_ensemble_algo.out"), true))
        out.println("iteration: " + i + " for given classifier, auPRC:" + raw)
        out.close
      }*/

      m
    }}.toList
    
    val ensemble = new NaiveBayesBinaryVoting(models)    
    val avg = ensemble.averageAuPRC(testData); val voted = ensemble.averageVotedAuPRC(testData)
    val m = NaiveBayes.train(baseData)
    val raw = (new BinaryClassificationMetrics(testData.map(p => (p.label,m.predict(p.features)))).areaUnderPR())
    val exp = ensemble.expertAuPRC(testData)
    
    println("num classifiers: " + numClassifiers)    
    println("average auPRC: " + avg)
    println("voted auPRC: " + voted)    
    println("expert auPRC: " + exp)
    println("raw auPRC: " + raw)
  }
}