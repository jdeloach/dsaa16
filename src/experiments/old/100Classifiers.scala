package experiments.old

import scala.collection.mutable.ListBuffer
import org.json4s._
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.linalg.Vectors
import scala.reflect.io.File
import org.apache.spark.mllib.classification.NaiveBayesModel

object Classifiers100 {
  def main(args : Array[String]) {
    val conf = new SparkConf()
      .setAppName("Classifier-averaging for Feature Selection on Unbalanced Datasets")
      //.setMaster("spark://129.130.10.134:7077")
      .setMaster("local[10]")
      .set("spark.driver.maxResultSize", "5g")
    val sc = new SparkContext(conf)  
    
    val thetas = generateThetas(sc)
    
    val featureWeightsPos = thetas.map { theta => theta(0) } // list->array->array->double ... model->class->feature->value
    val featureWeightsNeg = thetas.map { theta => theta(1) } // list->array->array->double ... model->class->feature->value
    
    //println("Num Classifiers: " + featureWeightsNeg.size + ", Num features: " + featureWeightsNeg(0).length)
    
    val toStats = sc.parallelize(featureWeightsNeg.map { x => Vectors.dense(x) })
    val toSingleStats = sc.parallelize(Seq(Vectors.dense(generateSingleTheta(sc)(0))))
    val meansAsc = List.fromArray(Statistics.colStats(toStats).mean.toArray).zipWithIndex.sortBy(_._1)
    val singleMeansAsc = List.fromArray(Statistics.colStats(toSingleStats).mean.toArray).zipWithIndex.sortBy(_._1)
    
    println(meansAsc.map(x => "(" + x._1 + "," + x._2 + ")").take(20))
    println(singleMeansAsc.map(x => "(" + x._1 + "," + x._2 + ")").take(20))
    //val statsByFeats = featsPos.map { x => Statistics.colStats(sc.parallelize(Seq(Vectors.dense(x.toArray)))) } // stats per feature
    //statsByFeats.foreach { x => println(x.mean) }
//    println((finalizedThetas(0).union(finalizedThetas(1))).sortBy { x => x }.mkString(" "))
    //[0][471] - average every spot and then print it
  }

  def loadThetas(sc: SparkContext) : List[Array[Array[Double]]] = {
    val models = List.fromArray(new java.io.File(".").listFiles()).filter { x => x.getName.startsWith("nb_") }
    models.map { x => NaiveBayesModel.load(sc, x.getAbsolutePath) }.map { x => x.theta }
  }

  def generateSingleTheta(sc: SparkContext) : Array[Array[Double]] = {
    val baseData = MLUtils.loadLibSVMFile(sc, "rq2_binaryClass.libsvm", 471, 6) ///Users/jdeloach/Code/Weka/weka-3-7-12/
    NaiveBayes.train(baseData, lambda = 1.0).theta
  }
  
  def generateThetas(sc: SparkContext) : List[Array[Array[Double]]] = {
    val partitions = 30
    val baseData = MLUtils.loadLibSVMFile(sc, "rq2_binaryClass.libsvm", 471, partitions).cache ///Users/jdeloach/Code/Weka/weka-3-7-12/
    val negativeSet = baseData.filter { x => x.label == 1 && x.features.numNonzeros > 0 }.collect()
    val positiveSet = List.fromArray(baseData.filter { x => x.label == 0 }.collect())
    
    val numClassifiers = (Math.floor(positiveSet.length / negativeSet.length)).toInt
    val positiveSets = positiveSet.grouped(negativeSet.length).toList
    val trainings = positiveSets.map { x => sc.parallelize(x.union(negativeSet)) }
    
    //trainings.foreach { x => println(x.collect().toList) }
    
    //println("# Neg Examples: " + negativeSet.length + ", # Pos Groups: " + positiveSets.length + ", Length of Ea. Pos Group: " + 
    //    positiveSets(0).length + " # classifiers: " + numClassifiers)
    //System.exit(0)
    var localVar = 0
    
    val thetas = trainings.map { training => {
      val model = NaiveBayes.train(training, lambda = 1.0)
      
      //model.save(sc, "nb_" + localVar + ".model")
      localVar += 1
            
      model.theta
    }}/*.reduce((a,b) => {
      val ret = Array.ofDim[Double](2,471)
      for(i <- 0 to 1) {
        for(j <- 0 to 470) {
          ret(i)(j) = a(i)(j) + b(i)(j)
        }
      }

      ret
    })*/
    
    thetas
  }
}