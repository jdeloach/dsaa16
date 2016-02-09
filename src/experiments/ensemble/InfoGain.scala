package experiments.old

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

object InfoGain {
  val f = new File("ml_diagnostics.txt")
  val nPartitions = 100
  val useSparse = false

  def main(args : Array[String]) {
    val conf = new SparkConf()
      .setAppName("Info Gain")
      //.setMaster("spark://129.130.10.134:7077")
      //.setMaster("local[6]")
      .set("spark.executor.memory", "120g")
      .set("spark.driver.maxResultSize", "35g")
    val sc = new SparkContext(conf) 
    val criterionStrings = List("mim", "mifs", "jmi", "mrmr", "icap", "cmim", "if")
    val criterion = new InfoThCriterionFactory("mim")
    val baseData = MLUtils.loadLibSVMFile(sc, "rq6_binaryClass.libsvm").repartition(nPartitions).cache ///Users/jdeloach/Code/Weka/weka-3-7-12/ //binaryClass
    val nToSelect = 100

    val splits = baseData.randomSplit(Array(0.67, 0.33))
    val sets = generateSets(sc, splits(0))
    
    
    criterionStrings.map { c => new InfoThCriterionFactory(c) }.foreach { c => execute(sc, sets, splits, baseData, c, nToSelect) }    
  }
  
  def execute(sc: SparkContext, sets: List[RDD[LabeledPoint]], splits:Array[RDD[LabeledPoint]], baseData:RDD[LabeledPoint], criterion: InfoThCriterionFactory, nToSelect: Int) : Unit = {
    val listOfTopFeatures = sets.par.map { data => {
      InfoThSelector2.train(criterion, 
              data, // RDD[LabeledPoint]
              nToSelect, // number of features to select
              nPartitions) // number of partitions
    }}.toList
    
    val perClassifierTop100Features = listOfTopFeatures.map(list => list.sortBy(x => Math.abs(x._2)).reverse.take(nToSelect).map(_._1).toArray)    
    val top100CommonFeats = listOfTopFeatures.flatten.groupBy{ x => x._1 }.map{case (idx,list) => (idx,list.map(x => Math.abs(x._2)).sum)}.toList.sortBy(f => f._2).reverse.take(nToSelect).map(_._1)
    val random100Feats = Random.shuffle(listOfTopFeatures.map(list => list.map(_._1).toArray).flatten.distinct.toList).take(nToSelect)
    val base100Feats = InfoThSelector.train(criterion, baseData, nToSelect, nPartitions).selectedFeatures

    val r1 = testNNaiveBayes(subsetOfFeatures(sets, top100CommonFeats), subsetOfFeatures(splits(1), top100CommonFeats), "Ensemble " + nToSelect + " Top (" + criterion.criterion + ")")
    val r2 = testNaiveBayes(subsetOfFeatures(splits(0), List.fromArray(base100Feats)), subsetOfFeatures(splits(1), List.fromArray(base100Feats)), "Single Classifier (" + nToSelect + ") features (" + criterion.criterion + ")")
//    val r3 = testNNaiveBayes(sets, splits(1), "Ensemble ALL Feats")
    val r4 = testNNaiveBayes(subsetOfFeatures(sets, random100Feats), subsetOfFeatures(splits(1), random100Feats), "Ensemble Random " + nToSelect + " Feats (" + criterion.criterion + ")")
    val r5 = testFeatureSpecificNNaiveBayes(subsetOfFeaturesPerClassifier(sets, perClassifierTop100Features), splits(1), perClassifierTop100Features, "Per Classifier Best " + nToSelect + " Features (" + criterion.criterion + ")")
    val r6 = testNNaiveBayes(subsetOfFeatures(sets, List.fromArray(base100Feats)), subsetOfFeatures(splits(1), List.fromArray(base100Feats)), "1-Classifier " + nToSelect + " Features used at the N-classifier Level (" + criterion.criterion + ")")
    
    diagnostics(r1 + "\n" + /*r3 +*/ "\n" + r4 + "\n" + r2 + "\n" + r5 + "\n" + r6)
    println(r1 + "\n" + /*r3 +*/ "\n" + r4 + "\n" + r2 + "\n" + r5 + "\n" + r6)
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
    assert(baseData.length == featureIdxsPerClassifier.length) // must have the same amount of datasets in each    
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
  
  /**
   * Prints out various auPRC metrics for a given SINGLE Naive Bayes instances.
   */
  def testNaiveBayes(train: RDD[LabeledPoint], test: RDD[LabeledPoint], testName: String) : String = {
      val model = NaiveBayes.train(train, lambda = 1.0)    
      val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
      //val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()
      val metrics = new BinaryClassificationMetrics(predictionAndLabel)
      "For test: " + testName + ", we got auROC: " + metrics.areaUnderROC() + " and auPRC: " + metrics.areaUnderPR()
  }
  
  /**
   * Uses N classifiers for a largely unbalanced dataset. N = #larger set/#smaller set. Prints out auRPC metrics.
   */
  def testNNaiveBayes(train: List[RDD[LabeledPoint]], test: RDD[LabeledPoint], testName: String) : String = {
    val classifiers = train.map { x => NaiveBayes.train(x) }
    val ensemble = new NaiveBayesBinaryVoting(classifiers)
    val predictionAndLabel = test.map(p => (ensemble.predict(p.features), p.label))
    val metrics = new BinaryClassificationMetrics(predictionAndLabel)
    "For test: " + testName + ", we got auROC: " + metrics.areaUnderROC() + " and auPRC: " + metrics.areaUnderPR() + ", expertAuPRC: " + 
      ensemble.expertAuPRC(test) //+ ", averageVotedAuPRC: " + ensemble.averageVotedAuPRC(test) + ", averageAuPRC: " + ensemble.averageAuPRC(test)
  }
  
  def testFeatureSpecificNNaiveBayes(train: List[RDD[LabeledPoint]], test: RDD[LabeledPoint], classifierFeatures: List[Array[Int]], testName: String) : String = {
    val classifiers = train.map { x => NaiveBayes.train(x) }
    val ensemble = new NBClassifierUniqueFeaturesEnsemble(classifiers.zip(classifierFeatures))
    val predictionAndLabel = test.map(p => (ensemble.predict(p.features), p.label))
    val metrics = new BinaryClassificationMetrics(predictionAndLabel)
    "For test: " + testName + ", we got auROC: " + metrics.areaUnderROC() + " and auPRC: " + metrics.areaUnderPR() + ", expertAuPRC: " + 
      ensemble.expertAuPRC(test) //+ ", averageVotedAuPRC: " + ensemble.averageVotedAuPRC(test) + ", averageAuPRC: " + ensemble.averageAuPRC(test)
  }
  
  def diagnostics(m: String) : Unit = {
    val pw = new PrintWriter(new FileOutputStream(f, true))
    pw.append(m + "\n")
    pw.close()
  }
}