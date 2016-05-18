package experiments.regularized.paper

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

object TemporalExperimentFScore {
  val f = new File("paper/6.4_diagnostics.txt")

  def main(args: Array[String]) : Unit = {
    val conf = new SparkConf()
      .setAppName("LR-LR Temporal Future")
      .set("spark.driver.maxResultSize", "45g")
    val sc = new SparkContext(conf) 
    val base = EnsembleUtils.loadBothLabelsDB(sc)
    val baseData = base.repartition(100).cache
      
    diagnostics(s"Starting LabelReg Class 0 Count: ${baseData.filter{_.label == 0}.count} Class 1 Count: ${baseData.filter{_.label == 1}.count}")
    diagnostics(s"2015 to 2016 Changes: ${baseData.filter { x => val y = x.asInstanceOf[MislabeledLabeledPoint]; y.realLabel != y.label }.count()}") 

    diagnostics(s"Including CHANGED from TRAINING")
    var folds = MLUtils.kFold(baseData, 5, 11).map{case (a,b) => (a.repartition(100).cache,b.repartition(100).cache)  }

    supervisedTests(folds)
    lrLr(folds)
    
    folds.foreach { case (a,b) => a.unpersist(false); b.unpersist(false) } // force spark to drop from memory ... just for safety
  }
  
  def supervisedTests(folds: Array[(RDD[LabeledPoint],RDD[LabeledPoint])]) {    
    val naiveBayesPRC = folds.map{ case (train,test) => {
      diagnostics(s"Number of Training Pos: ${train.filter { x => x.label == 1 }.count }, Neg: ${train.filter { x => x.label == 0 }.count}, Test Pos: ${test.filter { x => x.label == 1 }.count }, Neg: ${test.filter { x => x.label == 0 }.count}")
      
      val model = NaiveBayes.train(train)
      val predsAndLabel2015 = test.map{ x=> (model.predict(x.features), x.label) }
      val predsAndLabel2016 = test.map{ x=> (model.predict(x.features), x match { case m:MislabeledLabeledPoint => m.realLabel case _ => x.label }) }
      
      val mislabeled = test.filter { x => val y = x.asInstanceOf[MislabeledLabeledPoint]; y.realLabel != y.label }
      val mislabeledPreds = mislabeled.map { x => (model.predict(x.features),x.asInstanceOf[MislabeledLabeledPoint].realLabel) }
      (new BinaryClassificationMetrics(predsAndLabel2015, 100).areaUnderPR,EnsembleUtils.tpr(mislabeledPreds), EnsembleUtils.f1score(mislabeledPreds))
    }}.unzip3
    
    val lrPRC = folds.map{ case (train,test) => {
      val model = new LogisticRegressionWithLBFGS().run(train)
      val predsAndLabel2015 = test.map{ x=> (model.predict(x.features), x.label) }
      val predsAndLabel2016 = test.map{ x=> (model.predict(x.features), x match { case m:MislabeledLabeledPoint => m.realLabel case _ => x.label }) }
      
      val mislabeled = test.filter { x => val y = x.asInstanceOf[MislabeledLabeledPoint]; y.realLabel != y.label }
      val mislabeledPreds = mislabeled.map { x => (model.predict(x.features),x.asInstanceOf[MislabeledLabeledPoint].realLabel) }
      (new BinaryClassificationMetrics(predsAndLabel2015, 100).areaUnderPR,EnsembleUtils.tpr(mislabeledPreds), EnsembleUtils.f1score(mislabeledPreds))
    }}.unzip3

    
    val svmPRC = folds.map{ case (train,test) => {
      val model = SVMWithSGD.train(train, 200)
      val predsAndLabel2015 = test.map{ x=> (model.predict(x.features), x.label) }
      val predsAndLabel2016 = test.map{ x=> (model.predict(x.features), x match { case m:MislabeledLabeledPoint => m.realLabel case _ => x.label }) }
      
      val mislabeled = test.filter { x => val y = x.asInstanceOf[MislabeledLabeledPoint]; y.realLabel != y.label }
      val mislabeledPreds = mislabeled.map { x => (model.predict(x.features),x.asInstanceOf[MislabeledLabeledPoint].realLabel) }
      (new BinaryClassificationMetrics(predsAndLabel2015, 100).areaUnderPR,EnsembleUtils.tpr(mislabeledPreds), EnsembleUtils.f1score(mislabeledPreds))
    }}.unzip3

    
    diagnostics(s"NaiveBayes -- Test 2015 auPRC: ${naiveBayesPRC._1.sum / folds.size.toDouble} Test 2016 Changed TPR: ${naiveBayesPRC._2.sum / folds.size.toDouble}  2016 Changed F1 Score: ${naiveBayesPRC._3.sum / folds.size.toDouble}")
    diagnostics(s"LogisticRegression -- Test 2015 auPRC: ${lrPRC._1.sum / folds.size.toDouble} Test 2016 Changed TPR: ${lrPRC._2.sum / folds.size.toDouble} 2016 Changed F1 Score: ${lrPRC._3.sum / folds.size.toDouble}")
    diagnostics(s"SVM -- Test 2015 auPRC: ${svmPRC._1.sum / folds.size.toDouble} Test 2016 Changed TPR: ${svmPRC._2.sum / folds.size.toDouble} 2016 Changed F1 Score: ${svmPRC._3.sum / folds.size.toDouble}")
  }
  
  def lrLr(folds: Array[(RDD[LabeledPoint],RDD[LabeledPoint])]) {
    val pTildes = Array(0, .001, .01, .0322, .05)
    val lambdaUs = Array(1)
    
    pTildes.foreach{ pTilde =>
      lambdaUs.foreach { lambdaU =>
        val lrAvgAuPRC = folds.map{case (train,test) => experiment(train,test, pTilde, lambdaU)}.unzip3
        diagnostics(s"pTilde: " + pTilde + ", lambdaU: " + lambdaU + " Test2015 auPRC for LR: " + (lrAvgAuPRC._1.sum / folds.size.toDouble) + " 2016 Changed TPR: " + (lrAvgAuPRC._2.sum / folds.size.toDouble) + s" 2016 Changed F1 Score: ${lrAvgAuPRC._3.sum / folds.size.toDouble}")   
      }
    }
  }
  
  def experiment(train: RDD[LabeledPoint], test: RDD[LabeledPoint], pTilde: Double, lambdaU: Double) : (Double,Double,Double) = {
      val LRmodel = new LRLogisticRegressionWithLBFGS(pTilde, lambdaU).run(train)
      val predsAndLabel2015 = test.map{ x=> (LRmodel.predict(x.features), x.label) }
      val predsAndLabel2016 = test.map{ x=> (LRmodel.predict(x.features), x match { case m:MislabeledLabeledPoint => m.realLabel case _ => x.label }) }
      
      val mislabeled = test.filter { x => val y = x.asInstanceOf[MislabeledLabeledPoint]; y.realLabel != y.label }
      val mislabeledPreds = mislabeled.map { x => (LRmodel.predict(x.features),x.asInstanceOf[MislabeledLabeledPoint].realLabel) }
      (new BinaryClassificationMetrics(predsAndLabel2015, 100).areaUnderPR,EnsembleUtils.tpr(mislabeledPreds), EnsembleUtils.f1score(mislabeledPreds))
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