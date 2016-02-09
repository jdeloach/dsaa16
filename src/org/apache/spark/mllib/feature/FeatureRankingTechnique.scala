package org.apache.spark.mllib.feature

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.SVMWithSGD

abstract class FeatureRankingTechnique(data: RDD[LabeledPoint])  {  
  def featureValues : List[(Int,Double)]
}

class SVMWeightsRanking(data: RDD[LabeledPoint]) extends FeatureRankingTechnique(data) {
  private var weights = List.fromArray(SVMWithSGD.train(data, 100).weights.toArray).zipWithIndex.map{ case (value,index) => (index,value) }
  
  def featureValues = weights
}