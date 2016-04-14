package experiments.regularized

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
object ExperimentAndroidUnlabeled {
  val f = new File("labelreg_diagnostics.txt")

  def main(args: Array[String]) : Unit = {
    val conf = new SparkConf()
      .setAppName("Label Regularization")
      .setMaster("local[10]")
      //.set("spark.executor.memory", "35g")
      .set("spark.driver.maxResultSize", "45g")
    val sc = new SparkContext(conf) 
    //val (base,temporalTest) = EnsembleUtils.loadFromDB(sc)
    val baseData = EnsembleUtils.loadDenseArff(sc, "rq4_base_highquality_clean.arff", .05)._1.repartition(100).cache
    val (train,test) = baseData.randomSplit(Array(.67, .33), 11L) match { case Array(f1,f2) => (f1,f2) }
    val LRmodel = new LRLogisticRegressionWithLBFGS(.05, 1).run(train)
    val metrics = new BinaryClassificationMetrics(test.map { x => (LRmodel.predict(x.features),x.label) })
    println(s"auPRC: ${metrics.areaUnderPR}")
    
    /*val baseData = base.repartition(100).cache
    temporalTest.cache
    
    // .001 to .032 are the range we can do
    List(0, .001, .005, .01, .02, .032).foreach { noiseLevel => {
      val data = baseData //val data = redistributePosClass(baseData, noiseLevel) // (lowmal+benign), high_mal === (50+750),(50,000)
      diagnostics(s"Starting LabelReg.. TEST Noise Level: $noiseLevel Class 0 Count: ${data.filter{_.label == 0}.count} Class 1 Count: ${data.filter{_.label == 1}.count}")
      val folds = MLUtils.kFold(data, 3, 11).map{case (a,b) => (a.repartition(100).cache,redistributePosClass(b,noiseLevel).repartition(100).cache)  }
  
      supervisedTests(folds, temporalTest, noiseLevel)
      lrLr(folds, temporalTest, noiseLevel)
      
      folds.foreach { case (a,b) => a.unpersist(false); b.unpersist(false) } // force spark to drop from memory ... just for safety
    }}*/
  }
  
  /**
   * Take part of the positive class and relabel it as unlabeled so as to induce the scenario with unlabeled malware. (though noting that in the MislabeledLabeledPoint class).
   */
  def redistributePosClass(data: RDD[LabeledPoint], noiseLevel: Double): RDD[LabeledPoint] = {
    val unlabeledNegative = data.filter { _.label == 0 }.count
    val allPos = data.filter{_.label == 1}
    val numUnlabeledPos = (noiseLevel * unlabeledNegative) / (1 - noiseLevel)
      
    val labeledPos = .5;  // labeledPositive percentage of all positive
    val unlabeledPos = numUnlabeledPos / allPos.count.toDouble;  // unlabeledNegative percentage of all positive
    val rest = 1 - labeledPos - unlabeledPos; // throw away the rest
    
    val splits = allPos.randomSplit(Array(labeledPos, unlabeledPos, rest), 11L)
        
    data.filter{_.label == 0}.++(splits(1).map{ x => new MislabeledLabeledPoint(1,0,x.features) }).++(splits(0))    
  }
  
  def supervisedTests(folds: Array[(RDD[LabeledPoint],RDD[LabeledPoint])], temporalTest: RDD[LabeledPoint], noiseLevel: Double) {
    // NB, LR, SVM
    
    val naiveBayesPRC = folds.map{ case (train,test) => {
      diagnostics(s"Number of Training Pos: ${train.filter { x => x.label == 1 }.count }, Neg: ${train.filter { x => x.label == 0 }.count}, Test Pos: ${test.filter { x => x.label == 1 }.count }, Neg: ${test.filter { x => x.label == 0 }.count}")
      
      val model = NaiveBayes.train(train)
      val temporalTestAcc = temporalTest.filter{ x => model.predict(x.features) == 1 }.count / temporalTest.count.toDouble
      val x = test.first()
      val predsAndLabel = test.map{ x=> (model.predict(x.features), x match { case m:MislabeledLabeledPoint => m.realLabel case _ => x.label }) }
      val metrics = new BinaryClassificationMetrics(predsAndLabel, 100)
      val totalPos = test.filter{ x => x.label == 1 || (if(x.isInstanceOf[MislabeledLabeledPoint]) (x.asInstanceOf[MislabeledLabeledPoint]).realLabel == 1 else false) }
      val totalPosCorrect = totalPos.filter { x => model.predict(x.features) == 1 }.count.toDouble
      //savePRCurve(metrics.pr, s"naiveBayes")
      (metrics.areaUnderPR, temporalTestAcc, totalPosCorrect / totalPos.count.toDouble)
    }}.unzip3
    
    val lrPRC = folds.map{ case (train,test) => {
      val model = new LogisticRegressionWithLBFGS().run(train)
      val temporalTestAcc = temporalTest.filter{ x => model.predict(x.features) == 1 }.count / temporalTest.count.toDouble
      //model.clearThreshold 
      val predsAndLabel = test.map{ x=> (model.predict(x.features), x match { case m:MislabeledLabeledPoint => m.realLabel case _ => x.label }) }
      val metrics = new BinaryClassificationMetrics(predsAndLabel, 100)
      val actualPos = test.map{ x=> (model.predict(x.features),x.label) }.filter{ _._2 == 1 }
      val totalPos = test.filter{ x => x.label == 1 || (if(x.isInstanceOf[MislabeledLabeledPoint]) (x.asInstanceOf[MislabeledLabeledPoint]).realLabel == 1 else false) }
      val totalPosCorrect = totalPos.filter { x => model.predict(x.features) == 1 }.count.toDouble
      //savePRCurve(metrics.pr, s"logisticRegression")
      (metrics.areaUnderPR, temporalTestAcc, totalPosCorrect / totalPos.count.toDouble) 
    }}.unzip3
    
    val svmPRC = folds.map{ case (train,test) => {
      val model = SVMWithSGD.train(train, 200)
      val temporalTestAcc = temporalTest.filter{ x => model.predict(x.features) == 1 }.count / temporalTest.count.toDouble
      //model.clearThreshold
      val predsAndLabel = test.map{ x=> (model.predict(x.features), x match { case m:MislabeledLabeledPoint => m.realLabel case _ => x.label }) }
      val metrics = new BinaryClassificationMetrics(predsAndLabel, 100)
      val actualPos = test.map{ x=> (model.predict(x.features),x.label) }.filter{ _._2 == 1 }
      val totalPos = test.filter{ x => x.label == 1 || (if(x.isInstanceOf[MislabeledLabeledPoint]) (x.asInstanceOf[MislabeledLabeledPoint]).realLabel == 1 else false) }
      val totalPosCorrect = totalPos.filter { x => model.predict(x.features) == 1 }.count.toDouble
      //savePRCurve(metrics.pr, s"svmPRC")
      (metrics.areaUnderPR,temporalTestAcc, totalPosCorrect / totalPos.count.toDouble)
    }}.unzip3
    
    diagnostics(s"NaiveBayes -- Noise Level: $noiseLevel, auPRC: ${naiveBayesPRC._1.sum / folds.size.toDouble} Temporal Test ACC: ${naiveBayesPRC._2.sum / folds.size.toDouble}, POS-TOTAL ACC: ${naiveBayesPRC._3.sum / folds.size.toDouble}")
    diagnostics(s"LogisticRegression -- Noise Level: $noiseLevel, auPRC: ${lrPRC._1.sum / folds.size.toDouble} Temporal Test ACC: ${lrPRC._2.sum / folds.size.toDouble}, POS-TOTAL ACC: ${lrPRC._3.sum / folds.size.toDouble}")
    diagnostics(s"SVM -- Noise Level: $noiseLevel, auPRC: ${svmPRC._1.sum / folds.size.toDouble} Temporal Test ACC: ${svmPRC._2.sum / folds.size.toDouble}, POS-TOTAL ACC: ${svmPRC._3.sum / folds.size.toDouble}")
  }
  
  def lrLr(folds: Array[(RDD[LabeledPoint],RDD[LabeledPoint])], temporalTest: RDD[LabeledPoint], noiseLevel: Double) {
    val pTildes = Array(noiseLevel/*, noiseLevel + .01*/) // 25/775 50/800
    //val pTildes = Array(/*.001, .01,*/ .0322, .05, .0625, .1 /*, .2*/)
    val lambdaUs = Array(/*.5,*/ 1/*, 5, 10, 20, 50, 100, 1000, 10000, 50000, 100000, 500000, 1000000*/)
    
    pTildes.foreach{ pTilde =>
      lambdaUs.foreach { lambdaU =>
        val lrAvgAuPRC = folds.map{case (train,test) => experiment(train,test, pTilde, lambdaU, temporalTest)}.unzip3//.sum / folds.size.toDouble
        diagnostics(s"Noise Level: $noiseLevel, pTilde: " + pTilde + ", lambdaU: " + lambdaU + "auPRC for LR: " + (lrAvgAuPRC._1.sum / folds.size.toDouble) + " Temporal Test ACC for LR: " + (lrAvgAuPRC._2.sum / folds.size.toDouble) + s", POS-TOTAL ACC: ${lrAvgAuPRC._3.sum / folds.size.toDouble}")   
      }
    }
  }
  
  def experiment(train: RDD[LabeledPoint], test: RDD[LabeledPoint], pTilde: Double, lambdaU: Double, temporalTest: RDD[LabeledPoint]) : (Double,Double,Double) = {
    val LRmodel = new LRLogisticRegressionWithLBFGS(pTilde, lambdaU).run(train)
    val temporalTestAcc = temporalTest.filter{ x => LRmodel.predict(x.features) == 1 }.count / temporalTest.count.toDouble
    //LRmodel.clearThreshold
    val LRpredsAndLabel = test.map{ x=> (LRmodel.predict(x.features), x match { case m:MislabeledLabeledPoint => m.realLabel case _ => x.label }) }
    val lrMetrics = new BinaryClassificationMetrics(LRpredsAndLabel, 100)    
    //EnsembleUtils.printConfusionMatrix(List.fromArray(LRpredsAndLabel.collect()), 2)
    val actualPos = LRpredsAndLabel.filter{ _._2 == 1 }
    val totalPos = test.filter{ x => x.label == 1 || (if(x.isInstanceOf[MislabeledLabeledPoint]) (x.asInstanceOf[MislabeledLabeledPoint]).realLabel == 1 else false) }
    val totalPosCorrect = totalPos.filter { x => LRmodel.predict(x.features) == 1 }.count.toDouble
    //savePRCurve(lrMetrics.pr, s"labelReg_pTilde_$pTilde")
    
    savePRCurve(temporalTest.map{ x => (LRmodel.predict(x.features),1) }, s"temporalTest_lr$pTilde.csv")
    
    (lrMetrics.areaUnderPR(),temporalTestAcc, totalPosCorrect / totalPos.count.toDouble)
  }
  
  def savePRCurve(data: RDD[(Double,Double)], curveName: String) {
    if(new File(s"prc_curves/$curveName").exists) return
    val pw = new PrintWriter(new FileOutputStream(s"prc_curves/$curveName", false))
    data.collect.foreach{ case (a,b) => pw.append(s"$a,$b\n") }
    pw.close
  }
  
  /**
   * Undersamples the folds so that the training are 1 to 1, removing negative instances as necessary. Does not affect test sets.
   */
  def subsampleFolds(folds: Array[(RDD[LabeledPoint],RDD[LabeledPoint])]) : Array[(RDD[LabeledPoint],RDD[LabeledPoint])] = {        
    folds.map{ case (train,test) => {
      val negInstances = train.filter { x => x.label == 0 }; val posInstances = train.filter { x => x.label == 1 };
      (negInstances.sample(false, posInstances.count() / negInstances.count().toDouble, 11L) ++ posInstances,test)
    }}
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