package org.apache.spark.mllib.feature

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.SVMWithSGD

abstract class FeatureRankingTechnique {  
  var weights: List[(Int,Double)] = _

  def techniqueName = this.getClass.getName
  def train(data: RDD[LabeledPoint])
}

class SVMWeightsRanking extends FeatureRankingTechnique {
  
  def train(data: RDD[LabeledPoint]) : Unit = {
    weights = List.fromArray(SVMWithSGD.train(data, 100).weights.toArray).zipWithIndex.map{ case (value,index) => (index,value) }
  }  
}

class InfoThRanking(criterion: String) extends FeatureRankingTechnique {    
  def train(data: RDD[LabeledPoint]) : Unit = {
    weights = InfoThSelector2.train(new InfoThCriterionFactory(criterion), data, data.first().features.size, 100)
  }
  
  override def techniqueName = "InfoThRanking: " + criterion 
}