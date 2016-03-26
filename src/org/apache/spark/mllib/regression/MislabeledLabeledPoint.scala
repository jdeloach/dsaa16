package org.apache.spark.mllib.regression

import org.apache.spark.mllib.linalg.Vector

class MislabeledLabeledPoint(val realLabel: Double, misLabel: Double, features: Vector) extends LabeledPoint(misLabel, features) {}