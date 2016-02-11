package experiments.ensemble

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector

object EnsembleUtils {
  /** Returns the subset of features necessary for the provided model feature specification */
  def modelSpecificFeatureVector(modelFeatures: Array[Int], baseFeatureVector: Vector, useSparse:Boolean = false) : Vector = {
    if(useSparse)
      Vectors.sparse(baseFeatureVector.size, modelFeatures, modelFeatures.map { y => baseFeatureVector(y) }.toArray)
    else
      Vectors.dense(modelFeatures.map{ y => baseFeatureVector(y) }.toArray)
  }
}