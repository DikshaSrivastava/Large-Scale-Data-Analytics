package com.nuig.lsda;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

/**
 * The type K means clustering which uses Spark to cluster given Twitter tweets
 * by their geographic origins and counts the number of spam tweets in each cluster.
 *
 * @author Diksha Srivastava
 * @version 1.0
 */
public class KMeansClustering {

    /**
     * The entry point of application.
     *
     * @param args the input arguments
     */
    public static void main(String[] args) {

        // Setting up the spark configuration.
        SparkConf conf = new SparkConf().setAppName("K means clustering")
                .setMaster("local[4]")
                .set("spark.executor.memory", "1g");
        JavaSparkContext sc = new JavaSparkContext(conf);
        // Setting log level to error to show logs only when error comes.
        sc.setLogLevel("ERROR");

        // Reading the twitter file in Java RDD.
        String path = "./src/twitter2D_2.txt";
        JavaRDD<String> data = sc.textFile(path);

        /* Creating a Java Pair RDD containing features as key and a tuple of tweet and label as value.
           For example, a Java Pair RDD of the given form will be created:
           ([-56.544541,-29.089541],("Hey all this is what I did yesterday",0)) */
        JavaPairRDD<Vector, Tuple2<String, Integer>> parsedData = data.mapToPair(
                (String line) -> {
                    String[] tweetArray = line.split(",");
                    // Creating an array containing tweet's co-ordinates as features.
                    double[] features = new double[2];
                    features[0] = Double.parseDouble(tweetArray[2]);
                    features[1] = Double.parseDouble(tweetArray[3]);
                    return new Tuple2<>(Vectors.dense(features), new Tuple2<>(tweetArray[1], Integer.parseInt(tweetArray[0])));
                }
        );
        parsedData.cache();

        // Clustering the data into 4 classes using KMeans.
        int numClusters = 4;
        int numIterations = 20;
        // Training the model with all features.
        KMeansModel clusters = KMeans.train(parsedData.map(features -> features._1).rdd(), numClusters, numIterations);

        /* Allowing the model to predict on the trained model such that a new Java Pair RDD is created
           having cluster number in key and a tuple of tweet and label in value. */
        JavaPairRDD<Integer, Tuple2<String, Integer>> clusterAndTweet = parsedData
                .mapToPair(features -> new Tuple2<>(clusters.predict(features._1), features._2))
                .sortByKey(true, 1);
        // Cluster number is at _1 position of java pair RDD and tweet is at _2._1 position.
        clusterAndTweet.foreach(t -> System.out.println("Tweet "+t._2._1+" is in cluster "+t._1));

        // Creating a Java pair RDD of cluster and count of spam tweets i.e. having label 1.
        JavaPairRDD<Integer, Integer> clusterAndLabelCounts = clusterAndTweet
                .mapToPair(clusterAndLabel -> {
                    // Label is at position _2._2
                    if(clusterAndLabel._2._2 == 1)
                        // Cluster is at position ._1
                        return new Tuple2<>(clusterAndLabel._1, 1);
                    else
                        return new Tuple2<>(clusterAndLabel._1, 0);
                })
                .reduceByKey(Integer::sum)
                .sortByKey(true, 1);
        clusterAndLabelCounts
                .collect()
                .forEach(clusterAndLabel -> System.out.println("Cluster "+clusterAndLabel._1()
                        + " contains " + clusterAndLabel._2()+" spam tweets"));

        // Closing the Spark Context.
        sc.stop();
        sc.close();
    }
}
