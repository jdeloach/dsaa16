package experiments.ensemble

import java.io.File
import java.io.PrintWriter
import scala.annotation.elidable
import scala.annotation.elidable.ASSERTION
import scala.util.Random
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.InfoThCriterionFactory
import org.apache.spark.mllib.feature.InfoThSelector
import org.apache.spark.mllib.feature.InfoThSelector2
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import java.io.FileOutputStream
import scala.collection.immutable.HashMap
import scala.collection.mutable.ListBuffer
import org.apache.spark.mllib.feature.FeatureRankingTechnique
import org.apache.spark.mllib.feature.InfoThRanking
import org.apache.spark.mllib.feature.SVMWeightsRanking
import org.apache.spark.mllib.classification.SVMWithSGD

object InfoGain {
  val f = new File("ml_diagnostics.txt")
  val nPartitions = 100
  val useSparse = false

  def main(args : Array[String]) {
    val conf = new SparkConf()
      .setAppName("Info Gain")
      .setMaster("local[10]")
      .set("spark.executor.memory", "120g")
      .set("spark.driver.maxResultSize", "35g")
    val sc = new SparkContext(conf) 
    val techniques = List(new SVMWeightsRanking()/*, new InfoThRanking("mim"), new InfoThRanking("mifs"), new InfoThRanking("jmi"), new InfoThRanking("mrmr"), new InfoThRanking("icap"), new InfoThRanking("cmim"), new InfoThRanking("if")*/)
    val criterion = new InfoThCriterionFactory("mim")
    val baseData = MLUtils.loadLibSVMFile(sc, "rq2_quarter.libsvm").repartition(nPartitions).cache ///Users/jdeloach/Code/Weka/weka-3-7-12/ //binaryClass
    val nToSelect = 100
    val nFolds = 5 // 5-fold Cross Validation
    val folds = MLUtils.kFold(baseData, nFolds, 7)
    
    val sets = generateSets(sc, baseData)
    
    techniques.map { c => execute(sc, sets, baseData, c, nToSelect) }
  }
  
  def execute(sc: SparkContext, baseSets: List[RDD[LabeledPoint]], baseData:RDD[LabeledPoint], technique: FeatureRankingTechnique, nToSelect: Int) : Unit = {
    val listOfTopFeatures = baseSets.par.map { data => {
      technique.train(data)
      technique.weights
    }}.toList
    
    val perClassifierTop100Features = listOfTopFeatures.map(list => list.sortBy(x => Math.abs(x._2)).reverse.take(nToSelect).map(_._1).toArray)    
    val top100CommonFeats = listOfTopFeatures.flatten.groupBy{ x => x._1 }.map{case (idx,list) => (idx,list.map(x => Math.abs(x._2)).sum)}.toList.sortBy(f => f._2).reverse.take(nToSelect).map(_._1)
    val random100Feats = Random.shuffle(listOfTopFeatures.map(list => list.map(_._1).toArray).flatten.distinct.toList).take(nToSelect)
    technique.train(baseData);  val base100Feats = technique.weights.map(_._1)
    val resultsDB = new ListBuffer[(Int,Map[String,Any])] // [FOLD,[Key,Value]] -> key includes {test,auROC,auPRC,etc.}
    var fold = 1
    
    MLUtils.kFold(baseData, 5, 7).foreach{ case (train,test) => {
      val splits = Array(train,test)
      val sets = generateSets(sc, train)

       // NaiveBayes
      resultsDB += ((fold, testNNaiveBayes(subsetOfFeatures(sets, top100CommonFeats), subsetOfFeatures(splits(1), top100CommonFeats), "Ensemble " + nToSelect + " Top (" + technique.techniqueName + ")")));
      resultsDB += ((fold, testNaiveBayes(subsetOfFeatures(splits(0), base100Feats), subsetOfFeatures(splits(1), base100Feats), "Single Classifier (" + nToSelect + ") features (" + technique.techniqueName + ")")))
  //    val r3 = testNNaiveBayes(sets, splits(1), "Ensemble ALL Feats")
      resultsDB += ((fold, testNNaiveBayes(subsetOfFeatures(sets, random100Feats), subsetOfFeatures(splits(1), random100Feats), "Ensemble Random " + nToSelect + " Feats (" + technique.techniqueName + ")")));
// CAN't DO THIS IN 5-CV      val r5 = testFeatureSpecificNNaiveBayes(subsetOfFeaturesPerClassifier(sets, perClassifierTop100Features), splits(1), perClassifierTop100Features, "Per Classifier Best " + nToSelect + " Features (" + criterion.criterion + ")")
      resultsDB += ((fold, testNNaiveBayes(subsetOfFeatures(sets, base100Feats), subsetOfFeatures(splits(1), base100Feats), "1-Classifier " + nToSelect + " Features used at the N-classifier Level (" + technique.techniqueName + ")")))
      
      // SVM
      resultsDB += ((fold, testNSVM(subsetOfFeatures(sets, top100CommonFeats), subsetOfFeatures(splits(1), top100CommonFeats), "Ensemble " + nToSelect + " Top (" + technique.techniqueName + ")")));
      resultsDB += ((fold, testSVM(subsetOfFeatures(splits(0), base100Feats), subsetOfFeatures(splits(1), base100Feats), "Single Classifier (" + nToSelect + ") features (" + technique.techniqueName + ")")))
      resultsDB += ((fold, testNSVM(subsetOfFeatures(sets, random100Feats), subsetOfFeatures(splits(1), random100Feats), "Ensemble Random " + nToSelect + " Feats (" + technique.techniqueName + ")")));
      resultsDB += ((fold, testNSVM(subsetOfFeatures(sets, base100Feats), subsetOfFeatures(splits(1), base100Feats), "1-Classifier " + nToSelect + " Features used at the N-classifier Level (" + technique.techniqueName + ")")))
      
      fold += 1
      
      resultsDB
    }}
    
    val crossValidatedResults = resultsDB.groupBy{case (fold,map) => { map("test") }}.map { case (name:String,results) => {
      val metrics = results.map(_._2.keySet.filter { _ != "test" }).flatten.distinct
      (name,metrics.map( metric => (metric,results.map(_._2(metric).asInstanceOf[Double]).sum/fold)))
    }}
    
    val stringResults = crossValidatedResults.map{case (s,d) => s + ":" + d}.mkString("\n")
    diagnostics(stringResults)
    println(stringResults)
  }
  
  /**
   * Generates N different datasets with a 1:1 balance, assuming N times as many 0-class as 1-class examples.
   */
  def generateSets(sc: SparkContext, baseData: RDD[LabeledPoint]) : List[RDD[LabeledPoint]] = {
    val negativeSet = baseData.filter { x => x.label == 1 /*&& x.features.numNonzeros > 0*/ }.collect()
    val positiveSet = List.fromArray(baseData.filter { x => x.label == 0 }.collect())
    
    val numClassifiers = (Math.floor(positiveSet.length / negativeSet.length)).toInt
    val positiveSets = positiveSet.grouped(negativeSet.length).toList
    positiveSets.map { x => sc.parallelize(x.union(negativeSet)).cache() }
  }
  
  /**
   * Takes in a list of datasets, and a list of what features each of those datasets should be subsetted down to. Returns each dataset, with the correct per-dataset features.
   */
  def subsetOfFeaturesPerClassifier(baseData: List[RDD[LabeledPoint]], featureIdxsPerClassifier: List[Array[Int]]) : List[RDD[LabeledPoint]] = {
    assert(baseData.length == featureIdxsPerClassifier.length, "Number in Base Dataset (" + baseData.length + "), does not correlate with list of per-classifier features (" + featureIdxsPerClassifier.length + ").") // must have the same amount of datasets in each    
    baseData.zipWithIndex.map{ case (rdd,idx) => subsetOfFeatures(rdd, List.fromArray(featureIdxsPerClassifier(idx))) }
  }
  
  /**
   * Returns datasets that only use the provided subset of features for the different datasets provided.
   */
  def subsetOfFeatures(baseData: List[RDD[LabeledPoint]], featureIdxs: List[Int]) : List[RDD[LabeledPoint]] = {
    baseData.map{ rdd => subsetOfFeatures(rdd, featureIdxs) }
  }
  
  /**
   * Reduces the feature space from baseData to just those provided in featureIdxs.
   */
  def subsetOfFeatures(baseData: RDD[LabeledPoint], featureIdxs: List[Int]) : RDD[LabeledPoint] = {
    if(useSparse)
      baseData.map { x => new LabeledPoint(x.label,Vectors.sparse(x.features.size, featureIdxs.toArray, featureIdxs.map { y => x.features(y) }.toArray)) }
    else
      baseData.map { x => new LabeledPoint(x.label,Vectors.dense(featureIdxs.map { y => x.features(y) }.toArray)) }
  }
  
  def testSVM(train: RDD[LabeledPoint], test: RDD[LabeledPoint], testName: String) : Map[String,Any] = {
    val model = SVMWithSGD.train(train, 100)
      val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
      val metrics = new BinaryClassificationMetrics(predictionAndLabel)
      HashMap("test" -> ("SVM " + testName), "auROC" -> metrics.areaUnderROC(), "auPRC" -> metrics.areaUnderPR())    
  }

  /**
   * Uses N classifiers for a largely unbalanced dataset. N = #larger set/#smaller set. Prints out auRPC metrics.
   */
  def testNSVM(train: List[RDD[LabeledPoint]], test: RDD[LabeledPoint], testName: String) : Map[String,Any] = {
    val classifiers = train.map { x => SVMWithSGD.train(x, 100) }
    val ensemble = new LinearModelEnsemble(classifiers)
    val predictionAndLabel = test.map(p => (ensemble.predict(p.features), p.label))
    val metrics = new BinaryClassificationMetrics(predictionAndLabel)
    HashMap("test" -> ("NSVM " + testName), "auROC" -> metrics.areaUnderROC(), "auPRC" -> metrics.areaUnderPR()/*, "averageVotedAuPRC" -> ensemble.averageVotedAuPRC(test), "averageAuPRC" -> ensemble.averageAuPRC(test)*/)
  }
  
  /**
   * Prints out various auPRC metrics for a given SINGLE Naive Bayes instances.
   */
  def testNaiveBayes(train: RDD[LabeledPoint], test: RDD[LabeledPoint], testName: String) : Map[String,Any] = {
      val model = NaiveBayes.train(train, lambda = 1.0)    
      val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
      val metrics = new BinaryClassificationMetrics(predictionAndLabel)
      HashMap("test" -> ("NB " + testName), "auROC" -> metrics.areaUnderROC(), "auPRC" -> metrics.areaUnderPR())
  }
  
  /**
   * Uses N classifiers for a largely unbalanced dataset. N = #larger set/#smaller set. Prints out auRPC metrics.
   */
  def testNNaiveBayes(train: List[RDD[LabeledPoint]], test: RDD[LabeledPoint], testName: String) : Map[String,Any] = {
    val classifiers = train.map { x => NaiveBayes.train(x) }
    val ensemble = new NaiveBayesBinaryVoting(classifiers)
    val predictionAndLabel = test.map(p => (ensemble.predict(p.features), p.label))
    val metrics = new BinaryClassificationMetrics(predictionAndLabel)
    HashMap("test" -> ("NB " + testName), "auROC" -> metrics.areaUnderROC(), "auPRC" -> metrics.areaUnderPR(), "expertAuPRC" -> ensemble.expertAuPRC(test)/*, "averageVotedAuPRC" -> ensemble.averageVotedAuPRC(test), "averageAuPRC" -> ensemble.averageAuPRC(test)*/)
  }
  
  def testFeatureSpecificNNaiveBayes(train: List[RDD[LabeledPoint]], test: RDD[LabeledPoint], classifierFeatures: List[Array[Int]], testName: String) : Map[String,Any] = {
    val classifiers = train.map { x => NaiveBayes.train(x) }
    val ensemble = new NBClassifierUniqueFeaturesEnsemble(classifiers.zip(classifierFeatures))
    val predictionAndLabel = test.map(p => (ensemble.predict(p.features), p.label))
    val metrics = new BinaryClassificationMetrics(predictionAndLabel)
    HashMap("test" -> testName, "auROC" -> metrics.areaUnderROC(), "auPRC" -> metrics.areaUnderPR(), "expertAuPRC" -> ensemble.expertAuPRC(test)/*, "averageVotedAuPRC" -> ensemble.averageVotedAuPRC(test), "averageAuPRC" -> ensemble.averageAuPRC(test)*/)
  }
  
  def diagnostics(m: String) : Unit = {
    val pw = new PrintWriter(new FileOutputStream(f, true))
    pw.append(m + "\n")
    pw.close()
  }
}