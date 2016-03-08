package experiments.ensemble

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import scala.collection.mutable.ListBuffer

object EnsembleUtils {
  /** Returns the subset of features necessary for the provided model feature specification */
  def modelSpecificFeatureVector(modelFeatures: Array[Int], baseFeatureVector: Vector, useSparse:Boolean = false) : Vector = {
    if(useSparse)
      Vectors.sparse(baseFeatureVector.size, modelFeatures, modelFeatures.map { y => baseFeatureVector(y) }.toArray)
    else
      Vectors.dense(modelFeatures.map{ y => baseFeatureVector(y) }.toArray)
  }
  
  def loadArff(sc: SparkContext, fileName: String, sample: Double = 1.0) : (RDD[LabeledPoint], RDD[String]) = {
    val textfile = sc.textFile(fileName, 10)
   
    val features = sc.accumulableCollection(ListBuffer[String]())
    val instances = sc.accumulableCollection(ListBuffer[LabeledPoint]())
    
    textfile.foreach { x => {
      if(x.startsWith("@attribute")) {
        features += x.split(" ")(1) // title is second
      }
    }}
    
    val featureCount = features.value.size
    
    textfile.foreach { x => {
      if(x.startsWith("{") && x.length() > 2) {
        var (idx,value) = x.substring(1, x.length()-1).split(",").map { x => x.split(" ") }.filter(_(0) != "").map { x => (x(0).toInt,x(1)) }.toList.unzip
        val instanceClass = if (value.contains("malware")) 1 else 0
        instances += new LabeledPoint(instanceClass, Vectors.sparse(featureCount, idx.filter(_ != 0).toArray, value.filter(!_.equals("malware")).map { x => x.toDouble }.toArray))
      }
    }}
    
    (sc.parallelize(instances.value).sample(false, sample, 11L), sc.parallelize(features.value))
  }
  
  def printConfusionMatrix(predAndLabel: List[(Double,Double)], classCount: Int) = {
    // labelAndPred
    println(" " + (0 until classCount).mkString(" ") + " <- predicted ")
    println((0 until 2*classCount).map(x => "--").mkString)
    for(i <- 0 until classCount) {
      for(j <- 0 until classCount) {
        // i is actual
        // j is classified as
        print(" " + predAndLabel.count{ case (pred,label) => label == i && pred == j })
      }
      println(" | actual=" + i)
    }
  }
}