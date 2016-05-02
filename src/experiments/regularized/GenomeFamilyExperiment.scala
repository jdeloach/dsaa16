package experiments.regularized

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

object GenomeFamilyExperiment {
  val f = new File("6.5_diagnostics.txt")

  def main(args: Array[String]) : Unit = {
    val conf = new SparkConf()
      .setAppName("LR-LR  Genome Family Experiment")
      .set("spark.driver.maxResultSize", "45g")
    val sc = new SparkContext(conf) 
    val base = EnsembleUtils.load2012GenomeData(sc)
    val semiSupData = base(0)
    val supData = base(1)
    val unNegCount = semiSupData._1.filter { x => x.isInstanceOf[MislabeledLabeledPoint] }.count
    
    diagnostics(s"Genome Family Semi-Supervised Training Labeled Malware: ${semiSupData._1.filter { x => x.label == 1 }.count}, Training Unlabeled: ${semiSupData._1.filter { x => x.label == 0 }.count}, Training UnlabeledNegative: $unNegCount")
    diagnostics(s"Genome Family Supervised Training Labeled Malware: ${supData._1.filter { x => x.label == 1 }.count}, Training Negative: ${supData._1.filter { x => x.label == 0 }.count}")
    diagnostics(s"Genome Family TEST Size: ${supData._2.count}")
    
    supervisedTests(Array(supData))
    lrLr(Array(semiSupData))
  }
  
  def supervisedTests(folds: Array[(RDD[LabeledPoint],RDD[LabeledPoint])]) {    
    val naiveBayesPRC = folds.map{ case (train,test) => {      
      val model = NaiveBayes.train(train)
      val predsAndLabels = test.map{ x=> (model.predict(x.features), x.label) }
      (new BinaryClassificationMetrics(predsAndLabels, 100).areaUnderPR,EnsembleUtils.tpr(predsAndLabels))
    }}.unzip
    
    val lrPRC = folds.map{ case (train,test) => {
      val model = new LogisticRegressionWithLBFGS().run(train)
      val predsAndLabels = test.map{ x=> (model.predict(x.features), x.label) }
      (new BinaryClassificationMetrics(predsAndLabels, 100).areaUnderPR,EnsembleUtils.tpr(predsAndLabels))
    }}.unzip

    
    val svmPRC = folds.map{ case (train,test) => {
      val model = SVMWithSGD.train(train, 200)
      val predsAndLabels = test.map{ x=> (model.predict(x.features), x.label) }
      (new BinaryClassificationMetrics(predsAndLabels, 100).areaUnderPR,EnsembleUtils.tpr(predsAndLabels))
    }}.unzip

    
    diagnostics(s"NaiveBayes -- Test auPRC: ${naiveBayesPRC._1.sum / folds.size.toDouble} MAL TPR: ${naiveBayesPRC._2.sum / folds.size.toDouble}")
    diagnostics(s"LogisticRegression -- Test auPRC: ${lrPRC._1.sum / folds.size.toDouble} MAL TPR: ${lrPRC._2.sum / folds.size.toDouble}")
    diagnostics(s"SVM -- Test auPRC: ${svmPRC._1.sum / folds.size.toDouble} MAL TPR: ${svmPRC._2.sum / folds.size.toDouble}")
  }
  
  def lrLr(folds: Array[(RDD[LabeledPoint],RDD[LabeledPoint])]) {
    val pTildes = Array(/*.001, .01, .0322, */.05, .07, .1, .12)
    val lambdaUs = Array(1)
    
    pTildes.foreach{ pTilde =>
      lambdaUs.foreach { lambdaU =>
        val lrAvgAuPRC = folds.map{case (train,test) => experiment(train,test, pTilde, lambdaU)}.unzip
        diagnostics(s"pTilde: " + pTilde + ", lambdaU: " + lambdaU + " auPRC for LR-LR: " + (lrAvgAuPRC._1.sum / folds.size.toDouble) + " MAL TPR: " + (lrAvgAuPRC._2.sum / folds.size.toDouble))   
      }
    }
  }
  
  def experiment(train: RDD[LabeledPoint], test: RDD[LabeledPoint], pTilde: Double, lambdaU: Double) : (Double,Double) = {
      val LRmodel = new LRLogisticRegressionWithLBFGS(pTilde, lambdaU).run(train)
      val predsAndLabels = test.map{ x=> (LRmodel.predict(x.features), x.label) }
      (new BinaryClassificationMetrics(predsAndLabels, 100).areaUnderPR,EnsembleUtils.tpr(predsAndLabels))
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