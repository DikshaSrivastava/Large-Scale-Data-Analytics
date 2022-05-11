package com.nuig.lsda;

import org.apache.spark.SparkConf;
import org.apache.spark.streaming.Durations;
import org.apache.spark.streaming.api.java.JavaDStream;
import org.apache.spark.streaming.api.java.JavaPairDStream;
import org.apache.spark.streaming.api.java.JavaReceiverInputDStream;
import org.apache.spark.streaming.api.java.JavaStreamingContext;
import scala.Tuple2;

import java.util.Arrays;
import java.util.logging.Logger;

/**
 * The type Hashtag count class counts the hashtag and username in a sliding window of length 5 seconds
 * and a sliding interval of 2 seconds.
 *
 * @author Diksha Srivastava
 * @since 1.0
 */
public class HashtagCount_Q2 {

    private final static Logger LOGGER = Logger.getLogger(HashtagCount_Q2.class.getName());

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

        // Creating a DStream of words by splitting the tweets.
        JavaDStream<String> words = tweets.flatMap(tweet -> Arrays.asList(tweet.split(" ")).iterator());

        /* Creating a sliding window of 5 seconds and a sliding interval of 2 seconds and then filtering the
           hashtags/usernames and counting them using map-reduce concept. */
        JavaPairDStream<String, Integer> wordCounts = words.window(Durations.seconds(5),Durations.seconds(2))
                .filter(word -> word.startsWith("#") || word.startsWith("@"))
                .mapToPair(word -> new Tuple2<>(word, 1))
                .reduceByKey(Integer::sum);

        // Print the first ten elements of each RDD generated in this DStream.
        System.out.println("Count of hashtags and username: ");
        wordCounts.print();
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
