package org.apache.spark.mllib.optimization

import scala.math._
import breeze.linalg.{axpy => brzAxpy, norm => brzNorm, Vector => BV}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import breeze.linalg.{Vector => BV}
import breeze.linalg.{axpy => brzAxpy}
import breeze.linalg.{norm => brzNorm}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.commons.lang.NotImplementedException

class LabelRegularizedUpdater(pTilde:Double, lambdaU:Double) extends Updater {
  override def compute(
      weightsOld: Vector,
      gradient: Vector,
      stepSize: Double,
      iter: Int,
      regParam: Double): (Vector, Double) = { 
    throw new NotImplementedException("Use Compute() w/ data field for LabelRegularizedUpdater")
    (null,0d) 
  }

  def compute(
      weightsOld: Vector,
      gradient: Vector,
      stepSize: Double,
      iter: Int,
      regParam: Double,
      data: RDD[(Double,Vector)]): (Vector, Double) = {
        // add up both updates from the gradient of the loss (= step) as well as
        // the gradient of the regularizer (= regParam * weightsOld)
        // w' = w - thisIterStepSize * (gradient + regParam * w + KL-Gradient)
        val thisIterStepSize = stepSize / math.sqrt(iter)
        val brzWeights: BV[Double] = weightsOld.toBreeze.toDenseVector
        val grad = lrGradient(weightsOld,data)
        brzWeights -= thisIterStepSize * (gradient.toBreeze + (regParam * brzWeights) + grad._1)
        val norm = brzNorm(brzWeights, 2.0)
        (Vectors.fromBreeze(brzWeights), 0.5 * regParam * norm * norm + grad._2)
  }
  
  // vector and regularizer term @see compute() returns
  def lrGradient(weightsOld: Vector, data: RDD[(Double,Vector)]) : (BV[Double],Double) = {
    // TODO: implement minibatch fraction 1.0
    
    val currModel = new LogisticRegressionModel(weightsOld, 0, data.first()._2.size, 2)
    currModel.clearThreshold() // output probs
    val unlabeledClass = 0
    val unlabeled = data.sample(false, 1.0, 42).filter(_._1 == unlabeledClass)
    val unlabledCount = unlabeled.count
    val pThetaHat = 1/(unlabeled.count.toDouble) * unlabeled.map(data => currModel.predict(data._2)).sum
    
    val klDivergence = pTilde * math.log(1.0/pThetaHat) + (1 - pTilde) * math.log((1-pTilde)/(1-pThetaHat))
    val summation = unlabeled.map(data => data._2.toBreeze * (currModel.predict(data._2) * (1 - currModel.predict(data._2)))).reduce((v1,v2) => v1 + v2) // xi * p(y=1)(1-p(y=1)) for all unlabeled
    val lrGradient =  1 / (unlabledCount * 1.0) * ((1-pTilde)/(1-pThetaHat) - pTilde/pThetaHat) * summation
    
    (lambdaU * lrGradient, lambdaU * klDivergence)
  }
  
  def bhLrGradient(weightsOld: Vector, data: RDD[(Double,Vector)]) : (BV[Double],Double) = {
    // TODO: implement minibatch fraction 1.0
    
    val currModel = new LogisticRegressionModel(weightsOld, 0, data.first()._2.size, 2)
    currModel.clearThreshold() // output probs
    val unlabeledClass = 0
    val unlabeled = data.sample(false, 1.0, 42).filter(_._1 == unlabeledClass)
    val unlabledCount = unlabeled.count
    val pThetaHat = 1/(unlabeled.count.toDouble) * unlabeled.map(data => currModel.predict(data._2)).sum
    
    val bdDivergence = -1 * math.log(math.sqrt(pTilde*pThetaHat) + math.sqrt((1-pTilde)*(1-pThetaHat)))
    val summation = unlabeled.map(data => data._2.toBreeze * (currModel.predict(data._2) * (1 - currModel.predict(data._2)))).reduce((v1,v2) => v1 + v2) // xi * p(y=1)(1-p(y=1)) for all unlabeled
    val lrGradient =  -1/2.0 * (math.pow((pTilde-pThetaHat), -1/2.0)+math.pow((1-pTilde)*(1-pThetaHat), -1/2.0)) * summation / (math.sqrt(pTilde - pThetaHat) + math.sqrt((1-pTilde)*(1-pThetaHat)))
    
    (lambdaU * lrGradient, lambdaU * bdDivergence)
  }
}