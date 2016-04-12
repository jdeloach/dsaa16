package experiments.regularized.week4

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.util.MLUtils
import experiments.ensemble.EnsembleUtils
import java.io.File
import java.io.FileOutputStream
import java.io.PrintWriter
import java.text.SimpleDateFormat
import java.util.Date
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.classification.LRLogisticRegressionWithLBFGS
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.MislabeledLabeledPoint
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.classification.NaiveBayes

object HighConfGroundTruthExperiment {
  val f = new File("highConfGroundTruth4.2_diagnostics.txt")

  def main(args: Array[String]) : Unit = {
    val conf = new SparkConf()
      .setAppName("HighConfGroundTruth 4.2 LR LR")
      .set("spark.driver.maxResultSize", "45g")
    val sc = new SparkContext(conf) 
    
    val data = EnsembleUtils.loadHighConfAndRest(sc).repartition(100).cache
    diagnostics(s"Starting LabelReg Class 0 Count: ${data.filter{_.label == 0}.count} Class 1 Count: ${data.filter{_.label == 1}.count}")
    diagnostics(s"Noise (where 0<scannersCount<10): ${data.filter { x => val y = x.asInstanceOf[MislabeledLabeledPoint]; y.realLabel != y.label }.count()}") 
    val folds = MLUtils.kFold(data, 3, 11).map{case (a,b) => (a.repartition(100).cache,b.filter { x => x.asInstanceOf[MislabeledLabeledPoint].realLabel != -1 }.repartition(100).cache)  }
    
    supervisedTests(folds)
    lrLr(folds)
    
    folds.foreach { case (a,b) => a.unpersist(false); b.unpersist(false) } // force spark to drop from memory ... just for safety
  }
  
  def supervisedTests(folds: Array[(RDD[LabeledPoint],RDD[LabeledPoint])]) {
    // NB, LR, SVM
    
    val naiveBayesPRC = folds.map{ case (train,test) => {
      diagnostics(s"Number of Training Pos: ${train.filter { x => x.label == 1 }.count }, Neg: ${train.filter { x => x.label == 0 }.count}, Test Pos: ${test.filter { x => x.label == 1 }.count }, Neg: ${test.filter { x => x.label == 0 }.count}")
      
      val model = NaiveBayes.train(train)
      val predsAndLabel = test.map{ x=> (model.predict(x.features), x.asInstanceOf[MislabeledLabeledPoint].realLabel) }
      val metrics = new BinaryClassificationMetrics(predsAndLabel, 100)
      (metrics.areaUnderPR,EnsembleUtils.precision(predsAndLabel))
    }}.unzip
    
    val lrPRC = folds.map{ case (train,test) => {
      val model = new LogisticRegressionWithLBFGS().run(train)
      val predsAndLabel = test.map{ x=> (model.predict(x.features), x.asInstanceOf[MislabeledLabeledPoint].realLabel) }
      val metrics = new BinaryClassificationMetrics(predsAndLabel, 100)
      (metrics.areaUnderPR,EnsembleUtils.precision(predsAndLabel))
    }}.unzip

    val svmPRC = folds.map{ case (train,test) => {
      val model = SVMWithSGD.train(train, 200)
      val predsAndLabel = test.map{ x=> (model.predict(x.features), x.asInstanceOf[MislabeledLabeledPoint].realLabel) }
      val metrics = new BinaryClassificationMetrics(predsAndLabel, 100)
      (metrics.areaUnderPR,EnsembleUtils.precision(predsAndLabel))
    }}.unzip

    
    diagnostics(s"NaiveBayes -- auPRC: ${naiveBayesPRC._1.sum / folds.size.toDouble}, precision: ${naiveBayesPRC._2.sum / folds.size.toDouble}")
    diagnostics(s"LogisticRegression -- auPRC: ${lrPRC._1.sum / folds.size.toDouble}, precision: ${lrPRC._2.sum / folds.size.toDouble}")
    diagnostics(s"SVM -- auPRC: ${svmPRC._1.sum / folds.size.toDouble}, precision: ${svmPRC._2.sum / folds.size.toDouble}")
  }
  
  def lrLr(folds: Array[(RDD[LabeledPoint],RDD[LabeledPoint])]) {
    //val pTildes = Array(noiseLevel) // 25/775 50/800
    val pTildes = Array(.001, .01, .0322, .05, .0625, .1 , .2)
    val lambdaUs = Array(/*.5,*/ 1, 5,10/*, 20, 50, 100, 1000, 10000, 50000, 100000, 500000, 1000000*/)
    
    pTildes.foreach{ pTilde =>
      lambdaUs.foreach { lambdaU =>
        val lrAvgAuPRC = folds.map{case (train,test) => experiment(train,test, pTilde, lambdaU)}.unzip
        diagnostics(s"pTilde: " + pTilde + ", lambdaU: " + lambdaU + " auPRC for LR-LR: " + (lrAvgAuPRC._1.sum / folds.size.toDouble) + " precision: " + (lrAvgAuPRC._2.sum / folds.size.toDouble))   
      }
    }
  }
  
  def experiment(train: RDD[LabeledPoint], test: RDD[LabeledPoint], pTilde: Double, lambdaU: Double) : (Double,Double) = {
      val LRmodel = new LRLogisticRegressionWithLBFGS(pTilde, lambdaU).run(train)
      val predsAndLabel = test.map{ x=> (LRmodel.predict(x.features), x.asInstanceOf[MislabeledLabeledPoint].realLabel) }
      val metrics = new BinaryClassificationMetrics(predsAndLabel, 100)
      (metrics.areaUnderPR,EnsembleUtils.precision(predsAndLabel))
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