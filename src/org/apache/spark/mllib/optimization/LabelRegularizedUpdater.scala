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

class LabelRegularizedUpdater extends Updater {
  override def compute(
      weightsOld: Vector,
      gradient: Vector,
      stepSize: Double,
      iter: Int,
      regParam: Double): (Vector, Double) = { (null,0d) }
  
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
        // w' = (1 - thisIterStepSize * regParam) * w - thisIterStepSize * gradient
        val thisIterStepSize = stepSize / math.sqrt(iter)
        val brzWeights: BV[Double] = weightsOld.toBreeze.toDenseVector
        val grad = lrGradient(weightsOld,data)
        brzWeights -= thisIterStepSize * (gradient.toBreeze + (regParam * brzWeights) + grad._1)
        //brzWeights -= thisIterStepSize * lrGradient(weightsOld, data) // kl-divergence
        //brzWeights :*= (1.0 - thisIterStepSize * regParam)
//        if(weightsOld(0) != 0)
//          brzWeights += thisIterStepSize * (gradient.toBreeze + (regParam * lrGradient(weightsOld, data))) // iterStepSize * (gradient + (regParam * kl-Gradient))
        //brzAxpy(-thisIterStepSize, gradient.toBreeze, brzWeights)
        val norm = brzNorm(brzWeights, 2.0)
        (Vectors.fromBreeze(brzWeights), 0.5 * regParam * norm * norm + grad._2)
  }
  
  // vector and regularizer term
  def lrGradient(weightsOld: Vector, data: RDD[(Double,Vector)]) : (BV[Double],Double) = {
    // TODO: implement minibatch fraction 1.0
    
    val currModel = new LogisticRegressionModel(weightsOld, 0, data.first()._2.size, 2)
    currModel.clearThreshold() // output probs
    val unlabeledClass = 0
    val unlabeled = data.sample(false, 1.0, 42).filter(_._1 == unlabeledClass)
    val unlabledCount = unlabeled.count
    val pThetaHat = 1/(unlabeled.count.toDouble) * unlabeled.map(data => currModel.predict(data._2)).sum
    val pSwiggle = .01 // 8500/200000
    
    val klDivergence = pSwiggle * math.log(1.0/pThetaHat) + (1 - pSwiggle) * math.log((1-pSwiggle)/(1-pThetaHat))
    val summation = unlabeled.map(data => data._2.toBreeze * (currModel.predict(data._2) * (1 - currModel.predict(data._2)))).reduce((v1,v2) => v1 + v2) // xi * p(y=1)(1-p(y=1)) for all unlabeled
    val lambdaU = 10d//* data.filter(_._1 == 1).count // per Mann et. al in Mitchell
    val lrGradient =  1 / (unlabledCount * 1.0) * ((1-pSwiggle)/(1-pThetaHat) - pSwiggle/pThetaHat) * summation
    
    (lambdaU * lrGradient, lambdaU * klDivergence)
  }
}