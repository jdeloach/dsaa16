package experiments.regularized.paper

import java.io.File
import java.io.FileOutputStream
import java.io.PrintWriter
import java.text.SimpleDateFormat
import java.util.Date
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.annotation.Since
import org.apache.spark.mllib.classification.LRLogisticRegressionWithLBFGS
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import experiments.ensemble.EnsembleUtils
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.regression.MislabeledLabeledPoint

/**
 * @author jdeloach
 */
object VariedLambdasExperiment {
  val f = new File("paper/6.3_diagnostistics.txt")

  def main(args: Array[String]) : Unit = {
    val conf = new SparkConf()
      .setAppName("LR-LR VariedLambdasExperiment")
      .set("spark.driver.maxResultSize", "45g")
    val sc = new SparkContext(conf) 
    val baseData = EnsembleUtils.load2016LabelsSimple(sc).repartition(100).cache
    
    List(0).foreach { noiseLevel => {
      val data = baseData
      diagnostics(s"Starting LabelReg.. Mislabeled Noise Level: $noiseLevel Class 0 Count: ${data.filter{_.label == 0}.count} Class 1 Count: ${data.filter{_.label == 1}.count}")
      diagnostics(s"Starting LabelReg.. CORRECT Labeled Noise Level: $noiseLevel Class 0 Count: ${data.filter{x => x.isInstanceOf[MislabeledLabeledPoint] && x.asInstanceOf[MislabeledLabeledPoint].realLabel == 0}.count} Class 1 Count: ${data.filter{x => x.isInstanceOf[MislabeledLabeledPoint] && x.asInstanceOf[MislabeledLabeledPoint].realLabel == 1}.count}")
      val folds = MLUtils.kFold(data, 5, 11).map{case (a,b) => (a.repartition(100).cache,b.repartition(100).cache)  }
  
      supervisedTests(folds, noiseLevel)
      lrLr(folds, noiseLevel)
      
      folds.foreach { case (a,b) => a.unpersist(false); b.unpersist(false) } // force spark to drop from memory ... just for safety
    }}
  }
  
  def supervisedTests(folds: Array[(RDD[LabeledPoint],RDD[LabeledPoint])], noiseLevel: Double) {    
    val naiveBayesPRC = folds.map{ case (train,test) => {
      diagnostics(s"Number of Training Pos: ${train.filter { x => x.label == 1 }.count }, Neg: ${train.filter { x => x.label == 0 }.count}, Test Pos: ${test.filter { x => x.label == 1 }.count }, Neg: ${test.filter { x => x.label == 0 }.count}")
      
      val model = NaiveBayes.train(train)
      val predsAndLabel = test.map{ x=> (model.predict(x.features), x match { case m:MislabeledLabeledPoint => m.realLabel case _ => x.label }) }
      (new BinaryClassificationMetrics(predsAndLabel, 100).areaUnderPR)
    }}
    
    val lrPRC = folds.map{ case (train,test) => {
      val model = new LogisticRegressionWithLBFGS().run(train)
      val predsAndLabel = test.map{ x=> (model.predict(x.features), x match { case m:MislabeledLabeledPoint => m.realLabel case _ => x.label }) }
      (new BinaryClassificationMetrics(predsAndLabel, 100).areaUnderPR)
    }}
    
    val svmPRC = folds.map{ case (train,test) => {
      val model = SVMWithSGD.train(train, 200)
      val predsAndLabel = test.map{ x=> (model.predict(x.features), x match { case m:MislabeledLabeledPoint => m.realLabel case _ => x.label }) }
      (new BinaryClassificationMetrics(predsAndLabel, 100).areaUnderPR)
    }}
    
    diagnostics(s"NaiveBayes -- Noise Level: $noiseLevel, auPRC: ${naiveBayesPRC.sum / folds.size.toDouble}")
    diagnostics(s"LogisticRegression -- Noise Level: $noiseLevel, auPRC: ${lrPRC.sum / folds.size.toDouble}")
    diagnostics(s"SVM -- Noise Level: $noiseLevel, auPRC: ${svmPRC.sum / folds.size.toDouble}")
  }
  
  def lrLr(folds: Array[(RDD[LabeledPoint],RDD[LabeledPoint])], noiseLevel: Double) {
    val pTildes = Array(noiseLevel)
    val lambdaUs = Array(.5, 1, 5, 10, 20, 50, 100, 1000, 10000, 50000, 100000, 500000, 1000000)
    val regParams = Array(0, .00001, .0001, .001, .05, .5, 1, 5, 10, 20, 100, 500, 1000, 10000)
    
    regParams.foreach{ regParam => 
      pTildes.foreach{ pTilde =>
        lambdaUs.foreach { lambdaU =>
          val lrAvgAuPRC = folds.map{case (train,test) => experiment(train,test, pTilde, lambdaU, regParam)}//.sum / folds.size.toDouble
          diagnostics(s"Noise Level: $noiseLevel, pTilde: " + pTilde + s", regParam: $regParam, lambdaU: " + lambdaU + "auPRC for LR: " + (lrAvgAuPRC.sum / folds.size.toDouble))   
        }
      }
    }
  }
  
  def experiment(train: RDD[LabeledPoint], test: RDD[LabeledPoint], pTilde: Double, lambdaU: Double, regParam: Double) : Double = {
    val algo = new LRLogisticRegressionWithLBFGS(pTilde, lambdaU)
    algo.optimizer.setRegParam(regParam)
    val LRmodel = algo.run(train)
    val predsAndLabel = test.map{ x=> (LRmodel.predict(x.features), x match { case m:MislabeledLabeledPoint => m.realLabel case _ => x.label }) }
    (new BinaryClassificationMetrics(predsAndLabel, 100).areaUnderPR)
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