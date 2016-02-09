package org.arguslab.amandroid.ml

import scala.collection.mutable.ListBuffer
import org.json4s._
import org.json.JSONObject
import org.json.JSONArray

object CIS560 {
  def main(args : Array[String]) {
    val types = List("xaa", "xab", "xac", "xad", "xae", "xaf", "xag", "xah", "xai", "xaj")
    val file = "/Users/jdeloach/Dropbox/School/College/Junior Year/CIS 560/Project/images/new_an/"
    
    types.foreach { version => {

      println("parsing in raw text for " + version  +"...")
      val lines = scala.io.Source.fromFile(file + version).getLines.toList.map { _.replace("\\","").split(",", 2) }
      
      val sql = StringBuilder.newBuilder
      var i = 0
      
      lines.filter(_(1).startsWith("\"[")).foreach { row => {
        val appID = row(0).replace("\"", "")
        val jsonText = row(1).substring(1, row(1).length-1)
        val jsonArray = new JSONArray(jsonText)
        
        for(j <- 0 to jsonArray.length()-1) {
          val jsonObject = jsonArray.getJSONObject(j)
          val url = jsonObject.get("image_url")
          val options = if (jsonObject.optBoolean("supports_fife_url_options", false)) 1 else 0
          val sequence = if(jsonObject.optInt("position_in_sequence", -1) == -1) "" else jsonObject.optInt("position_in_sequence", -1).toString()
          val imageType = jsonObject.getInt("image_type") 
          
          sql.append("INSERT IGNORE INTO `playDrone`.`AppImages` VALUES ('" + appID + "','" + url + "','" + options + "','" + sequence + "','" + imageType + "');\n")
          i += 1
          
         if(i % 1000 == 0) {
          println("Status: " + version + " " + i)
          scala.tools.nsc.io.File("appimages" + version + ".sql").appendAll(sql.toString)
          sql.clear()
         }
        }}
      }
      scala.tools.nsc.io.File("appimages" + version + ".sql").appendAll(sql.toString)
      sql.clear()
    }}
  }

  def perms() {
    val file = "/Users/jdeloach/Dropbox/School/College/Junior Year/CIS 560/Project/permissions.csv"
    val lines = scala.io.Source.fromFile(file).getLines.toList.map { _.replace("\"", "").split(",", 2) }
    
    val sql = StringBuilder.newBuilder
    
    println("Size: " + lines.size)
    println("First row: " + lines(0)(0) + "," + lines(0)(1))
    var i = 0
        
    lines.foreach { row => {
      row(1).split(",").filter { p => p.forall { _.isDigit } && p != "" && p.toInt <= 139 }.foreach { p => sql.append("insert ignore into playDrone.App_Permission VALUES(" + p + ", \"" + row(0)  +"\");\n")  }
      i += 1
      
      if(i % 100000 == 0) {
        println("Status: " + i)
        scala.tools.nsc.io.File("insertPermsRelationship.sql").appendAll(sql.toString)
        sql.clear()
      }
    }}
    
    println("Done!")
  }
}