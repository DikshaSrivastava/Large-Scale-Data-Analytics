package com.nuig.lsda;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.streaming.Durations;
import org.apache.spark.streaming.api.java.JavaReceiverInputDStream;
import org.apache.spark.streaming.api.java.JavaStreamingContext;
import scala.Tuple2;

import java.util.Arrays;
import java.util.logging.Logger;

/**
 * The type Hashtag frequency calculates the frequency of every hashtag and username relatively.
 *
 * @author Diksha Srivastava
 * @since 1.0
 */
public class HashtagFrequency_Q4 {

    private final static Logger LOGGER = Logger.getLogger(HashtagFrequency_Q4.class.getName());

    /**
     * The entry point of application.
     *
     * @param args the input arguments
     */
    public static void main(String[] args) {

        // Create a local StreamingContext with two working thread and batch interval of 1 second.
        SparkConf conf = new SparkConf().setMaster("local[2]").setAppName("TweetCount");
        JavaStreamingContext jssc = new JavaStreamingContext(conf, Durations.seconds(1));
        // Setting the logger level of streaming context to error.
        jssc.sparkContext().setLogLevel("ERROR");

        // Creating a DStream that will connect to localhost:9999.
        JavaReceiverInputDStream<String> tweets = jssc.socketTextStream("localhost", 9999);
        tweets.window(Durations.seconds(5),Durations.seconds(2));

        // Calculating the relative frequency of every hashtag.
        System.out.println("Relative frequency of hashtags: ");
        tweets.foreachRDD(rdd -> {
                    // Finding the total count of all words in a window.
                    long totalCount = rdd.flatMap(tweet -> Arrays.asList(tweet.split(" ")).iterator()).count();
                    // Finding the count of hashtag and usernames using map-reduce.
                    JavaPairRDD<String, Double> hashTagCount = rdd.flatMap(tweet ->
                                    Arrays.asList(tweet.split(" ")).iterator())
                            .filter(word -> word.startsWith("#") || word.startsWith("@"))
                            .mapToPair(word -> new Tuple2<>(word, 1.0))
                            .reduceByKey(Double::sum);
                    hashTagCount.collect().forEach(hashCount -> System.out.println("("+hashCount._1+", "+
                            String.format("%.4f",hashCount._2/totalCount)+")"));
                }
        );

        // Starting the computation.
        jssc.start();
        try {
            // Waiting for the computation to terminate.
            jssc.awaitTermination();
        } catch (InterruptedException e) {
            LOGGER.info("Execution interrupted: "+ e.getMessage());
        }
        jssc.close();
    }
}
