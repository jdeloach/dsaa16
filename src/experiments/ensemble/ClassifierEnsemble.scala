package experiments.ensemble

import scala.annotation.elidable.ASSERTION
import scala.collection.mutable.ArrayBuffer

import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.classification.NaiveBayesModel
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.InfoThRanking
import org.apache.spark.mllib.feature.SVMWeightsRanking
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.GeneralizedLinearModel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD


trait ClassifierEnsemble {
  def predict(features: Vector) : Double
  def averageAuPRC(test: RDD[LabeledPoint]) : Double
}

class NaiveBayesBinaryVoting(models: List[NaiveBayesModel]) extends ClassifierEnsemble with Serializable {
  val useSparse = false
  
  def predict(features: Vector) : Double = {
    models.groupBy(_.predict(features)).maxBy(_._2.length)._1    
  }
    
  def predictProbabilities(features: Vector) : Vector = {
    models.map { x => x.predictProbabilities(features) }.reduce((v1,v2) => addVectors(v1,v2))
  }
  
  /**
   * Takes the auPRC of the ensemble classifier where each subclassifier sums its certainty towards each label
   * This is different from voting as these votes are weighted with regards to certainty.
   */
  def expertAuPRC(test: RDD[LabeledPoint]) : Double = {
    val predictionAndLabel = test.map { point => {
      val pred = predictProbabilities(point.features)
      if(pred(0) > pred(1)) (0.0,point.label) else (1.0,point.label)
    }}
    
    val metrics = new BinaryClassificationMetrics(predictionAndLabel)
    metrics.areaUnderPR()
  }
  
  /**
   * Computes auPRC for each classifier, and then averages the auPRCs
   */
  def averageAuPRC(test: RDD[LabeledPoint]) : Double = {
    val auPRCs = models.map { model => {
      // Evaluate model on test instances and compute test error
      val predictionAndLabel = test.map { point =>
        val prediction = model.predict(point.features)
        (prediction, point.label)
      }
      
      val metrics = new BinaryClassificationMetrics(predictionAndLabel)
      metrics.areaUnderPR()
    }}
    
    auPRCs.sum / auPRCs.length
  }
  
  /**
   * Takes the auPRC of the ensemble classifier created with voting of subclassifiers
   */
  def averageVotedAuPRC(test: RDD[LabeledPoint]) : Double = {
    val predictionAndLabel = test.map { point =>
      val prediction = predict(point.features)
      (prediction,point.label)
    }
        
    val metrics = new BinaryClassificationMetrics(predictionAndLabel)
    metrics.areaUnderPR()    
  }
  
  protected def addVectors(v1: Vector, v2: Vector) : Vector = {
    val a1 = v1.toArray; val a2 = v2.toArray;
    val b = ArrayBuffer[Double]()

    assert(a1.length == 2 && a2.length == 2)
    
    for(i <- 0 to a1.length-1) {
      b += a1(i) + a2(i)
    }
    
    Vectors.dense(b.toArray)
  }
}

class NBClassifierUniqueFeaturesEnsemble(models: List[(NaiveBayesModel,Array[Int])]) extends NaiveBayesBinaryVoting(null) with Serializable {
  override def predict(features: Vector) : Double = {
    models.groupBy{ case (model,mFeats) => model.predict(EnsembleUtils.modelSpecificFeatureVector(mFeats,features))}.maxBy(_._2.length)._1    
  }
    
  override def predictProbabilities(features: Vector) : Vector = {
    models.map { case (model,mFeats) => model.predictProbabilities(EnsembleUtils.modelSpecificFeatureVector(mFeats,features)) }.reduce((v1,v2) => addVectors(v1,v2))
  }
  
  /**
   * Computes auPRC for each classifier, and then averages the auPRCs
   */
  override def averageAuPRC(test: RDD[LabeledPoint]) : Double = {
    val auPRCs = models.map { case (model,mFeats) => {
      // Evaluate model on test instances and compute test error
      val predictionAndLabel = test.map { point =>
        val prediction = model.predict(EnsembleUtils.modelSpecificFeatureVector(mFeats,point.features))
        (prediction, point.label)
      }
      
      val metrics = new BinaryClassificationMetrics(predictionAndLabel)
      metrics.areaUnderPR()
    }}
    
    auPRCs.sum / auPRCs.length
  }
}

class LinearModelEnsemble(models: List[GeneralizedLinearModel]) extends ClassifierEnsemble with Serializable {
  def predict(features: Vector) : Double = {
    models.groupBy(_.predict(features)).maxBy(_._2.length)._1    
  }
  
  /**
   * Computes auPRC for each classifier, and then averages the auPRCs
   */
  def averageAuPRC(test: RDD[LabeledPoint]) : Double = {
    val auPRCs = models.map { model => {
      // Evaluate model on test instances and compute test error
      val predictionAndLabel = test.map { point =>
        val prediction = model.predict(point.features)
        (prediction, point.label)
      }
      
      val metrics = new BinaryClassificationMetrics(predictionAndLabel)
      metrics.areaUnderPR()
    }}
    
    auPRCs.sum / auPRCs.length
  }
  
  /**
   * Takes the auPRC of the ensemble classifier created with voting of subclassifiers
   */
  def averageVotedAuPRC(test: RDD[LabeledPoint]) : Double = {
    val predictionAndLabel = test.map { point =>
      val prediction = predict(point.features)
      (prediction,point.label)
    }
        
    val metrics = new BinaryClassificationMetrics(predictionAndLabel)
    metrics.areaUnderPR()    
  }
}

class HeterogeneousFeatureSubsetClassifierEnsemble(sets: List[RDD[LabeledPoint]]) extends ClassifierEnsemble with Serializable {
  private var coreEnsemble : NBClassifierUniqueFeaturesEnsemble = _
  
  private def init() : Unit = {
    val techniques = List(new SVMWeightsRanking(), new InfoThRanking("mim"), new InfoThRanking("mifs"), new InfoThRanking("jmi"), new InfoThRanking("mrmr")/*, new InfoThRanking("icap"), new InfoThRanking("cmim"), new InfoThRanking("if")*/)

    val listOfTopFeatures = sets.par.map { data => {
      val technique = techniques(scala.util.Random.nextInt(techniques.size))
      technique.train(data)
      technique.weights.map(_._1).sortBy { x => x }.reverse.take(100).toArray
    }}.toList
    
    val cleanedData = listOfTopFeatures.zip(sets).map{ case (features,set) => set.map { point => new LabeledPoint(point.label,EnsembleUtils.modelSpecificFeatureVector(features, point.features)) }}
    val models = cleanedData.map { train => NaiveBayes.train(train) }.toList
    coreEnsemble = new NBClassifierUniqueFeaturesEnsemble(models.zip(listOfTopFeatures))
  }
  
  init()
  
  def predict(features: Vector) : Double = coreEnsemble.predict(features)
  def predictProbabilities(features: Vector) : Vector = coreEnsemble.predictProbabilities(features)
  def averageAuPRC(test: RDD[LabeledPoint]) : Double = coreEnsemble.averageAuPRC(test)
  def averageVotedAuPRC(test: RDD[LabeledPoint]) : Double = coreEnsemble.averageVotedAuPRC(test)
  def expertAuPRC(test: RDD[LabeledPoint]) : Double = coreEnsemble.expertAuPRC(test)
}