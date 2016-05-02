package experiments.regularized

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import experiments.ensemble.EnsembleUtils
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors

object SplitFolds {
  def main(args: Array[String]) : Unit = {
    val conf = new SparkConf()
      .setAppName("LR InjectedNoiseExperiment")
            .setMaster("local[10]")

      .set("spark.driver.maxResultSize", "45g")
    val sc = new SparkContext(conf) 
    val baseData = addID(EnsembleUtils.load2016LabelsSimple(sc)).cache

    println(s"done loading data in. row size: ${baseData.count}")
    val threeFold = MLUtils.kFold(baseData, 3, 11)
    val fiveFold = MLUtils.kFold(baseData, 5, 11)
    save(threeFold)
    save(fiveFold)
  }
  
  def addID(points: RDD[LabeledPoint]) = points.zipWithUniqueId().map { case (x,id) => new LabeledPoint(x.label, Vectors.dense(Array(id.toDouble) ++ x.features.toArray)) }
  
  def save(folds:Array[(RDD[LabeledPoint],RDD[LabeledPoint])]) : Unit = {
    for(i <- 1 to folds.size) {
      MLUtils.saveAsLibSVMFile(folds(i-1)._1, s"/media/disks/hdd/Jordan/folds_out/fold-$i-of-${folds.size}_train.libsvm")
      MLUtils.saveAsLibSVMFile(folds(i-1)._2, s"/media/disks/hdd/Jordan/folds_out/fold-$i-of-${folds.size}_test.libsvm")
    }  
  }
}