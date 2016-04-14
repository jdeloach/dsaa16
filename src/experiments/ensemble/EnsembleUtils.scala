package experiments.ensemble

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import scala.collection.mutable.ListBuffer
import org.apache.spark.sql.Row
import org.apache.spark.mllib.regression.MislabeledLabeledPoint

object EnsembleUtils {
  val mysqlURL = scala.io.Source.fromInputStream(getClass.getResourceAsStream("/resources/mysql.txt")).bufferedReader().lines().findFirst().get

  /** Returns the subset of features necessary for the provided model feature specification */
  def modelSpecificFeatureVector(modelFeatures: Array[Int], baseFeatureVector: Vector, useSparse:Boolean = false) : Vector = {
    if(useSparse)
      Vectors.sparse(baseFeatureVector.size, modelFeatures, modelFeatures.map { y => baseFeatureVector(y) }.toArray)
    else
      Vectors.dense(modelFeatures.map{ y => baseFeatureVector(y) }.toArray)
  }
  
  def loadArff(sc: SparkContext, fileName: String, sample: Double = 1.0) : (RDD[LabeledPoint], RDD[String]) = {
    val textfile = sc.textFile(fileName, 10)
   
    val features = sc.accumulableCollection(ListBuffer[String]())
    val instances = sc.accumulableCollection(ListBuffer[LabeledPoint]())
    
    textfile.foreach { x => {
      if(x.startsWith("@attribute")) {
        features += x.split(" ")(1) // title is second
      }
    }}
    
    val featureCount = features.value.size
    
    textfile.foreach { x => {
      if(x.startsWith("{") && x.length() > 2) {
        var (idx,value) = x.substring(1, x.length()-1).split(",").map { x => x.split(" ") }.filter(_(0) != "").map { x => (x(0).toInt,x(1)) }.toList.unzip
        val instanceClass = if (value.contains("malware")) 1 else 0
        instances += new LabeledPoint(instanceClass, Vectors.sparse(featureCount, idx.filter(_ != 0).toArray, value.filter(!_.equals("malware")).map { x => x.toDouble }.toArray))
      }
    }}
    
    (sc.parallelize(instances.value).sample(false, sample, 11L), sc.parallelize(features.value))
  }
  
  /**
   * Returns the base/general dataset, containing positive and negative points.
   * Also contains a special forward-temporal dataset.
   */
  def loadFromDB(sc: SparkContext) : (RDD[LabeledPoint],RDD[LabeledPoint]) = {
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

    val jdbcDF = sqlContext.read.format("jdbc").options(
      Map("url" -> mysqlURL,
      "dbtable" -> "appSec")).load()
      
    val testSet = jdbcDF.filter("class = 'playdrone_benign' and scannersCount > 9")
                  .map { x => { val point = rowToLabeledPoint(x); new LabeledPoint(1d, point.features) }}    
    val baseSet = jdbcDF.filter("class = 'playdrone_benign' or class = 'high_mal'").map { rowToLabeledPoint(_) }.filter { _.label != -1 }
    println(s"Test Dataset Size: ${testSet.count}")
    (baseSet,testSet)
  }
  
  /**
   * Returns MislabeledPoints 
   */
  def loadBothLabelsDB(sc: SparkContext) : (RDD[LabeledPoint]) = {
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

    val jdbcDF = sqlContext.read.format("jdbc").options(
      Map("url" -> mysqlURL,
      "dbtable" -> "appSec")).load()
      
    jdbcDF.filter("class = 'playdrone_benign' or class = 'high_mal'").map { rowToMislabeledPoint(_) }.filter { x => x.label != -1 && x.asInstanceOf[MislabeledLabeledPoint].realLabel != -1  }   
  }
  
  /**
   * Loads High Confidence (>=10) as Positive, and REST as Negative/Unlabeled. E.g. Count=5 would be unlabeled/negative
   */
  def loadHighConfAndRest(sc: SparkContext) : RDD[LabeledPoint] = {
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

    val jdbcDF = sqlContext.read.format("jdbc").options(
      Map("url" -> mysqlURL,
      "dbtable" -> "appSec")).load()    
    
    jdbcDF.filter("scannersCount != -1").map { row => {
      val trainLabel = row.getInt(row.fieldIndex("scannersCount")) match {
        case x if x >= 20 => 1d
        case x if x < 20 && x >= 0 => 0d
        case _ => -1d
      }
      val testLabel = row.getInt(row.fieldIndex("scannersCount")) match {
        case x if x >= 10 => 1d
        case 0 => 0d
        case _ => -1d
      }
      
      val features = Vectors.dense(row.toSeq.drop(4).map { _.asInstanceOf[Integer].toDouble }.toArray)
      new MislabeledLabeledPoint(testLabel, trainLabel,features).asInstanceOf[LabeledPoint]
    }}.filter { x => x.label != -1 }       
  }
  
  def rowToMislabeledPoint(row: Row) : LabeledPoint = {    
    new MislabeledLabeledPoint(rowToLabeledPoint(row), row.getInt(row.fieldIndex("scannersCount")) match {
      case x if x > 9 => 1d
      case x if x == 0 => 0d
      case _ => -1d
    })
  }
  
  def rowToLabeledPoint(row: Row) : LabeledPoint = {
    val label = row.getString(row.fieldIndex("class")) match {
      case "playdrone_benign" => 0d
      case "high_mal" => 1d
      case _ => -1d
    }
    val features = Vectors.dense(row.toSeq.drop(4).map { _.asInstanceOf[Integer].toDouble }.toArray)
    
    new LabeledPoint(label,features)    
  }
  
  /**
   * @param maxBenign is the maximum the app can have to be benign, e.g. 0 for 0 scanners max can consider it benign. 1 for 0 or 1
   * @param minMalware the minimum scanners to consider it malware. E.g. 10 if you want >=10
   */
  def loadScannerCounts(sc:SparkContext, maxBenign:Int, minMalware:Int) : RDD[LabeledPoint] = {
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

    val jdbcDF = sqlContext.read.format("jdbc").options(
      Map("url" -> mysqlURL,
      "dbtable" -> "appSec")).load()    
    
    jdbcDF.filter("scannersCount != -1").map { row => {
      val label = row.getInt(row.fieldIndex("scannersCount")) match {
        case x if x >= minMalware => 1d
        case x if x <= maxBenign => 0d
        case _ => -1d
      }
      val features = Vectors.dense(row.toSeq.drop(4).map { _.asInstanceOf[Integer].toDouble }.toArray)
      new LabeledPoint(label,features)
    }}.filter { x => x.label != -1 }      
  }
  
  def loadDenseArff(sc: SparkContext, fileName:String, sample: Double = 1.0) : (RDD[LabeledPoint], RDD[String]) = {
    val textfile = sc.textFile(fileName, 10)
   
    val features = sc.accumulableCollection(ListBuffer[String]())
    val instances = sc.accumulableCollection(ListBuffer[LabeledPoint]())
    
    textfile.foreach { x => {
      if(x.startsWith("@attribute")) {
        features += x.split(" ")(1) // title is second
      }
    }}
    
    textfile.foreach { x => {
      if(!x.startsWith("@") && x.length() > 2) {
        val splits = x.split(',')
        val instanceClass = if (splits(0) == "high_mal"/* || splits(1) == "low_mal"*/) 1 else 0
        val featureVector = Vectors.dense(splits.tail.map { x => x.toDouble }.toArray)
        
        if(splits(0) != "low_mal")
          instances += new LabeledPoint(instanceClass, featureVector)
      }
    }}
    
    (sc.parallelize(instances.value).sample(false, sample, 11L), sc.parallelize(features.value))
  }
  
  def tpr(predsAndLabel: RDD[(Double,Double)]) : Double = {
    val tp = predsAndLabel.filter(x => x._1 == x._2 && x._2 == 1).count; 
    val p = predsAndLabel.count; 
    tp / p.toDouble
  }
  
  def precision(predsAndLabel: RDD[(Double,Double)]) : Double = {
    val tp = predsAndLabel.filter(x => x._1 == x._2 && x._2 == 1).count; 
    val fp = predsAndLabel.filter(x => x._1 == 1 && x._2 == 0).count; 
    tp / (tp + fp).toDouble
  }
  
  def f1score(predsAndLabel: RDD[(Double,Double)]) : Double = {
    val precision = EnsembleUtils.precision(predsAndLabel)
    val recall = EnsembleUtils.tpr(predsAndLabel)
    2 * (precision * recall) / (precision + recall)
  }
    
  def printConfusionMatrix(predAndLabel: List[(Double,Double)], classCount: Int) = {
    // labelAndPred
    println(" " + (0 until classCount).mkString(" ") + " <- predicted ")
    println((0 until 2*classCount).map(x => "--").mkString)
    for(i <- 0 until classCount) {
      for(j <- 0 until classCount) {
        // i is actual
        // j is classified as
        print(" " + predAndLabel.count{ case (pred,label) => label == i && pred == j })
      }
      println(" | actual=" + i)
    }
  }
}