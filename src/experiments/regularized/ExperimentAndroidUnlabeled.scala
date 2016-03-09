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
import java.io.FileOutputStream
import java.io.PrintWriter
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date


/**
 * @author jdeloach
 */
object ExperimentAndroidUnlabeled {
  val f = new File("labelreg_diagnostics.txt")

  def main(args: Array[String]) : Unit = {
    val conf = new SparkConf()
      .setAppName("Label Regularization")
      //.setMaster("local[10]")
      //.set("spark.executor.memory", "35g")
      .set("spark.driver.maxResultSize", "45g")
    val sc = new SparkContext(conf) 
    val data = MLUtils.loadLibSVMFile(sc, "rq2_binaryClass.libsvm")
    val folds = MLUtils.kFold(data, 3, 11)
    
    val pTildes = Array(.001, .01, .05, .1, .2)
    val lambdaUs = Array(/*.5, 1, 5, 10, 20, 50, 100, 1000, */10000, 50000, 100000, 500000, 1000000)

    val lrAvgAuPRC = folds.map{case (train,test) => {
      val regModel = new LogisticRegressionWithLBFGS().run(train)
      val regPredsAndLabel = test.map { x => (regModel.predict(x.features),x.label) }
      val regMetrics = new BinaryClassificationMetrics(regPredsAndLabel)
      regMetrics.areaUnderPR()
    }}.sum / folds.size.toDouble
    
    diagnostics("auPRC for Reg: " + lrAvgAuPRC / folds.size.toDouble)
    
    for(pTilde <- pTildes) {
      for(lambdaU <- lambdaUs) {
        val lrAvgAuPRC = folds.map{case (train,test) => experiment(train,test, pTilde, lambdaU)}.sum / folds.size.toDouble
        diagnostics("pTilde: " + pTilde + ", lambdaU: " + lambdaU + " auPRC for LR: " + lrAvgAuPRC)   
      }
    }
  }
  
  def experiment(train: RDD[LabeledPoint], test: RDD[LabeledPoint], pTilde: Double, lambdaU: Double) : (Double) = {
    val LRmodel = new LRLogisticRegressionWithLBFGS(pTilde, lambdaU).run(train)
    val LRpredsAndLabel = test.map { x => (LRmodel.predict(x.features),x.label) }
    val lrMetrics = new BinaryClassificationMetrics(LRpredsAndLabel)    
    //EnsembleUtils.printConfusionMatrix(List.fromArray(LRpredsAndLabel.collect()), 2)
    
    (lrMetrics.areaUnderPR())
  }
  
  def diagnostics(m: String, date:Boolean = true) : Unit = {
    val pw = new PrintWriter(new FileOutputStream(f, true))
    if(date)
      pw.append(new SimpleDateFormat("MM/dd/yyyy HH:mm:ss").format(new Date()) + " " + m + "\n")
    else
      pw.append(m + "\n")
    pw.close()
  }
}