package com.nuig.lsda;

import org.apache.spark.*;
import org.apache.spark.streaming.*;
import org.apache.spark.streaming.api.java.*;

import java.util.logging.Logger;

/**
 * The type Tweets which receive tweets from Twotter and prints it.
 *
 * @author Diksha Srivastava
 * @since 1.0
 */
public class Tweets_Q1 {

    private final static Logger LOGGER = Logger.getLogger(Tweets_Q1.class.getName());

    /**
     * The entry point of application.
     *
     * @param args the input arguments
     */
    public static void main(String[] args) {

        // Create a local StreamingContext with two working thread and batch interval of 2 seconds.
        SparkConf conf = new SparkConf().setMaster("local[2]").setAppName("TweetCount");
        JavaStreamingContext jssc = new JavaStreamingContext(conf, Durations.seconds(2));
        // Setting the logger level of streaming context to error.
        jssc.sparkContext().setLogLevel("ERROR");

        // Creating a DStream that will connect to localhost:9999.
        JavaReceiverInputDStream<String> tweets = jssc.socketTextStream("localhost", 9999);

        // Printing the first ten elements of each RDD generated in this DStream.
        System.out.println("Tweets: ");
        tweets.print();
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
