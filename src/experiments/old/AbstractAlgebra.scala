package org.arguslab.amandroid.ml

object AbstractAlgebra {
  def main(args: Array[String]) {
    var a = 0.0
    var b = 0
    
    for(a <- -1000 to 1000) {
      for (b <- -1000 to 1000) {
        if(a != 0 && b != 0) {
          var result = b*b / a
          if(Math.floor(result) == result) {
            result = Math.pow(b, 4).toInt / Math.pow(a, 2).toInt
            if(Math.floor(result) != result) {
              println("Proved wrong with a: " + a + ", b:" + b)
            }
          } else {
            println("Assumption wrong")
          }
        }
      }
    }
  }
}