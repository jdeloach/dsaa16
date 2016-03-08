package experiments.regularized

import org.apache.spark.mllib.classification.LRLogisticRegressionWithSGD
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import experiments.ensemble.EnsembleUtils
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.mllib.classification.LRLogisticRegressionWithLBFGS
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint


/**
 * @author jdeloach
 */
object ExperimentAndroidUnlabeled {
  def main(args: Array[String]) : Unit = {
    val conf = new SparkConf()
      .setAppName("Label Regularization")
      //.setMaster("local[10]")
      //.set("spark.executor.memory", "35g")
      .set("spark.driver.maxResultSize", "45g")
    val sc = new SparkContext(conf) 
    val data = MLUtils.loadLibSVMFile(sc, "rq2_binaryClass.libsvm")
    val folds = MLUtils.kFold(data, 3, 11)
    
    val (lrAvgAuPRC,regAvgAuPRC) = folds.map{case (train,test) => experiment(train,test)}.reduce((a1,a2) => (a1._1+a2._1,a1._2+a2._2))
    
    val output = new ArrayBuffer[String]()
    
    output += "auPRC for LR: " + lrAvgAuPRC / folds.size.toDouble
    output += "auPRC for Reg: " + regAvgAuPRC / folds.size.toDouble
    
    println(output.mkString("\n"))
  }
  
  def experiment(train: RDD[LabeledPoint], test: RDD[LabeledPoint]) : (Double,Double) = {
    val LRmodel = new LRLogisticRegressionWithLBFGS().run(train)//LRLogisticRegressionWithSGD.train(sets(0), 150)
    val LRpredsAndLabel = test.map { x => (LRmodel.predict(x.features),x.label) }
    val lrMetrics = new BinaryClassificationMetrics(LRpredsAndLabel)
    
    val regModel = new LogisticRegressionWithLBFGS().run(train)
    val regPredsAndLabel = test.map { x => (regModel.predict(x.features),x.label) }
    val regMetrics = new BinaryClassificationMetrics(regPredsAndLabel)
    
    //EnsembleUtils.printConfusionMatrix(List.fromArray(LRpredsAndLabel.collect()), 2)
    //EnsembleUtils.printConfusionMatrix(List.fromArray(regPredsAndLabel.collect()), 2)  
    
    (lrMetrics.areaUnderPR(), regMetrics.areaUnderPR())
  }
}