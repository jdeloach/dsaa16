package experiments.ensemble

import java.io.File
import java.io.FileOutputStream
import java.io.PrintWriter
import java.text.SimpleDateFormat
import java.util.Date
import scala.collection.immutable.HashMap
import scala.collection.mutable.ListBuffer
import scala.util.Random
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.FeatureRankingTechnique
import org.apache.spark.mllib.feature.InfoThCriterionFactory
import org.apache.spark.mllib.feature.SVMWeightsRanking
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.feature.InfoThRanking

object InfoGain {
  val f = new File("ml_diagnostics.txt")
  val nPartitions = 100
  val useSparse = false
  val sample = .2
  val nFolds = 3 // 3-fold Cross Validation

  def main(args : Array[String]) {
    val conf = new SparkConf()
      .setAppName("Info Gain")
      //.setMaster("local[10]")
      //.set("spark.executor.memory", "35g")
      .set("spark.driver.maxResultSize", "45g")
    val sc = new SparkContext(conf) 
        
    val (data,features) = EnsembleUtils.loadArff(sc, "drebin.arff", sample)
    val skipPrefixes = List("activity::", "intent::", "provider::", "service_receiver::", "url::")
    val featsToKeep = features.zipWithIndex().filter{ case (feature:String,index:Long) => !skipPrefixes.map { x => feature.startsWith(x) }.reduce(_ || _) }.collect    
    val subsetData = data.map { x => new LabeledPoint(x.label, EnsembleUtils.modelSpecificFeatureVector(featsToKeep.map(_._2.toInt), x.features, true)) }
    val nToSelect = (featsToKeep.size * .05).toInt
    diagnostics("Beginning InfoGain Experiments with Drebin data. Original feature count: " + features.count + ", Subsampled Feature Count: " + nToSelect + ", Overall Data Sample: " + sample, true)
    
    // JMI is the one that causes us to often fail.
    val techniques = List(new SVMWeightsRanking()/*, new InfoThRanking("mim"), new InfoThRanking("mifs"), /*new InfoThRanking("jmi"),*/ new InfoThRanking("mrmr"), new InfoThRanking("icap"), new InfoThRanking("cmim"), new InfoThRanking("if")*/)
    val criterion = new InfoThCriterionFactory("mim")
//    val baseData = MLUtils.loadLibSVMFile(sc, "drebin.libsvm").repartition(nPartitions).sample(false, .5, 11).cache ///Users/jdeloach/Code/Weka/weka-3-7-12/ //binaryClass

    val sets = generateSets(sc, subsetData)
    
    techniques.map { c => execute(sc, sets, subsetData, c, nToSelect) }
  }
  
  def execute(sc: SparkContext, baseSets: List[RDD[LabeledPoint]], baseData:RDD[LabeledPoint], technique: FeatureRankingTechnique, nToSelect: Int) : Unit = {
    val listOfTopFeatures = baseSets.par.map { data => {
      technique.train(data)
      technique.weights
    }}.toList
    diagnostics("Finished using " + technique.techniqueName + " to rank features...", true)
    
    val perClassifierTop100Features = listOfTopFeatures.map(list => list.sortBy(x => Math.abs(x._2)).reverse.take(nToSelect).map(_._1).toArray)    
    val top100CommonFeats = listOfTopFeatures.flatten.groupBy{ x => x._1 }.map{case (idx,list) => (idx,list.map(x => Math.abs(x._2)).sum)}.toList.sortBy(f => f._2).reverse.take(nToSelect).map(_._1)
    val random100Feats = Random.shuffle(listOfTopFeatures.map(list => list.map(_._1).toArray).flatten.distinct.toList).take(nToSelect)
    technique.train(baseData);  val base100Feats = technique.weights.map(_._1)
    val resultsDB = new ListBuffer[(Int,Map[String,Any])] // [FOLD,[Key,Value]] -> key includes {test,auROC,auPRC,etc.}
    var fold = 1
    diagnostics("Finished calculating various subsets of " + technique.techniqueName + " features.", true)
    
    MLUtils.kFold(baseData, nFolds, 7).par.foreach{ case (train,test) => {
      val splits = Array(train,test)
      val sets = generateSets(sc, train)

       // NaiveBayes
      resultsDB += ((fold, testNNaiveBayes(subsetOfFeatures(sets, top100CommonFeats), subsetOfFeatures(splits(1), top100CommonFeats), "Ensemble " + nToSelect + " Top (" + technique.techniqueName + ")")));
      resultsDB += ((fold, testNaiveBayes(subsetOfFeatures(splits(0), base100Feats), subsetOfFeatures(splits(1), base100Feats), "Single Classifier (" + nToSelect + ") features (" + technique.techniqueName + ")")))
  //    val r3 = testNNaiveBayes(sets, splits(1), "Ensemble ALL Feats")
      resultsDB += ((fold, testNNaiveBayes(subsetOfFeatures(sets, random100Feats), subsetOfFeatures(splits(1), random100Feats), "Ensemble Random " + nToSelect + " Feats (" + technique.techniqueName + ")")));
      //resultsDB += ((fold, testFeatureSpecificNNaiveBayes(subsetOfFeaturesPerClassifier(sets, perClassifierTop100Features), splits(1), perClassifierTop100Features, "Per Classifier Best " + nToSelect + " Features (" + technique.techniqueName + ")")))
      resultsDB += ((fold, testNNaiveBayes(subsetOfFeatures(sets, base100Feats), subsetOfFeatures(splits(1), base100Feats), "1-Classifier " + nToSelect + " Features used at the N-classifier Level (" + technique.techniqueName + ")")))
      resultsDB += ((fold, testNaiveBayes(subsetOfFeatures(splits(0), top100CommonFeats), subsetOfFeatures(splits(1), top100CommonFeats), "Single Classifier (" + nToSelect + ") with ENSEMBLE-SELECTED features (" + technique.techniqueName + ")")))
     
      // SVM
      resultsDB += ((fold, testNSVM(subsetOfFeatures(sets, top100CommonFeats), subsetOfFeatures(splits(1), top100CommonFeats), "Ensemble " + nToSelect + " Top (" + technique.techniqueName + ")")));
      resultsDB += ((fold, testSVM(subsetOfFeatures(splits(0), base100Feats), subsetOfFeatures(splits(1), base100Feats), "Single Classifier (" + nToSelect + ") features (" + technique.techniqueName + ")")))
      resultsDB += ((fold, testNSVM(subsetOfFeatures(sets, random100Feats), subsetOfFeatures(splits(1), random100Feats), "Ensemble Random " + nToSelect + " Feats (" + technique.techniqueName + ")")));
      resultsDB += ((fold, testNSVM(subsetOfFeatures(sets, base100Feats), subsetOfFeatures(splits(1), base100Feats), "1-Classifier " + nToSelect + " Features used at the N-classifier Level (" + technique.techniqueName + ")")))
      resultsDB += ((fold, testSVM(subsetOfFeatures(splits(0), top100CommonFeats), subsetOfFeatures(splits(1), top100CommonFeats), "Single Classifier (" + nToSelect + ") with ENSEMBLE-SELECTED features (" + technique.techniqueName + ")")))
      

      // NHeterogenouslySelectedFeatures
      //resultsDB += ((fold, testNHeterogenouslySelectedFeaturesNaiveBayes(sets, splits(1), "Heterogenous Feature Subsets Per Classifier (" + technique.techniqueName + ")")))

      diagnostics("Fold " + fold + "/" + nFolds + "  of " + technique.techniqueName + " completed.", true)
      
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

  def testNHeterogenouslySelectedFeaturesNaiveBayes(train: List[RDD[LabeledPoint]], test: RDD[LabeledPoint], testName: String) : Map[String,Any] = {
      val ensemble = new HeterogeneousFeatureSubsetClassifierEnsemble(train)
      val predictionAndLabel = test.map(p => (ensemble.predict(p.features), p.label))
      val metrics = new BinaryClassificationMetrics(predictionAndLabel)
      HashMap("test" -> ("NB " + testName), "auROC" -> metrics.areaUnderROC(), "auPRC" -> metrics.areaUnderPR(), "expertAuPRC" -> ensemble.expertAuPRC(test))
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
  
  def diagnostics(m: String, date:Boolean = false) : Unit = {
    val pw = new PrintWriter(new FileOutputStream(f, true))
    if(date)
      pw.append(new SimpleDateFormat("MM/dd/yyyy HH:mm:ss").format(new Date()) + " " + m + "\n")
    else
      pw.append(m + "\n")
    pw.close()
  }
}