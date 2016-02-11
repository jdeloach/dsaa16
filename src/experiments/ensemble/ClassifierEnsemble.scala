package experiments.ensemble

import scala.annotation.elidable
import scala.annotation.elidable.ASSERTION
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.mllib.classification.NaiveBayesModel
import org.apache.spark.mllib.classification.SVMModel
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.regression.GeneralizedLinearModel


trait ClassifierEnsemble {
  def predict(features: Vector) : Double
  def predictProbabilities(features: Vector) : Vector
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
    models.groupBy{ case (model,mFeats) => model.predict(modelSpecificFeatureVector(mFeats,features))}.maxBy(_._2.length)._1    
  }
    
  override def predictProbabilities(features: Vector) : Vector = {
    models.map { case (model,mFeats) => model.predictProbabilities(modelSpecificFeatureVector(mFeats,features)) }.reduce((v1,v2) => addVectors(v1,v2))
  }
  
  /**
   * Computes auPRC for each classifier, and then averages the auPRCs
   */
  override def averageAuPRC(test: RDD[LabeledPoint]) : Double = {
    val auPRCs = models.map { case (model,mFeats) => {
      // Evaluate model on test instances and compute test error
      val predictionAndLabel = test.map { point =>
        val prediction = model.predict(modelSpecificFeatureVector(mFeats,point.features))
        (prediction, point.label)
      }
      
      val metrics = new BinaryClassificationMetrics(predictionAndLabel)
      metrics.areaUnderPR()
    }}
    
    auPRCs.sum / auPRCs.length
  }
  
  /** Returns the subset of features necessary for the provided model feature specification */
  private def modelSpecificFeatureVector(modelFeatures: Array[Int], baseFeatureVector: Vector) : Vector = {
    if(useSparse)
      Vectors.sparse(baseFeatureVector.size, modelFeatures, modelFeatures.map { y => baseFeatureVector(y) }.toArray)
    else
      Vectors.dense(modelFeatures.map{ y => baseFeatureVector(y) }.toArray)
  }
}

class LinearModelEnsemble(models: List[GeneralizedLinearModel]) extends ClassifierEnsemble with Serializable {
  def predict(features: Vector) : Double = {
    models.groupBy(_.predict(features)).maxBy(_._2.length)._1    
  }
  
  def predictProbabilities(features: Vector) : Vector = {
    throw new NotImplementedError("LinearModelEnsembles do not support posterior probabilities.")
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
