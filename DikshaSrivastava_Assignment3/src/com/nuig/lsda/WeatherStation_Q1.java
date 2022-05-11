package com.nuig.lsda;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * The type Weather station which finds the count of temperatures across all stations.
 *
 * @author Diksha Srivastava
 * @since 1.0
 */
public class WeatherStation_Q1 implements Serializable {

    /**
     * The constant stations which stores all the existing weather stations.
     */
    public static List<WeatherStation_Q1> stations = new ArrayList<>();

    private String city;
    private List<Measurement_Q1> measurements;

    /**
     * Gets city.
     *
     * @return the city
     */
    public String getCity() {
        return city;
    }

    /**
     * Sets city.
     *
     * @param city the city
     */
    public void setCity(String city) {
        this.city = city;
    }

    /**
     * Gets measurements.
     *
     * @return the measurements
     */
    public List<Measurement_Q1> getMeasurements() {
        return measurements;
    }

    /**
     * Sets measurements.
     *
     * @param measurements the measurements
     */
    public void setMeasurements(List<Measurement_Q1> measurements) {
        this.measurements = measurements;
    }

    /**
     * Count temperatures returns the count of temperature t across all stations.
     *
     * @param t the temperature
     * @return the count of temperature t.
     */
    public static int countTemperatures(double t) {

        // Adds hadoop home environment variable.
        System.setProperty("hadoop.home.dir", "C:/winutils");

        // Setting up the spark configuration with 4 threads.
        SparkConf sparkConf = new SparkConf()
                .setAppName("TemperatureCount")
                .setMaster("local[4]")
                .set("spark.executor.memory", "1g");
        JavaSparkContext ctx = new JavaSparkContext(sparkConf);
        // Setting log level to error to show logs only when error comes.
        ctx.setLogLevel("ERROR");

        // Creating java RDD from existing stations using parallel computing.
        JavaRDD<WeatherStation_Q1> weatherStations = ctx.parallelize(stations);
        //Creating a java RDD of Measurement from weatherStations.
        JavaRDD<Measurement_Q1> measurement = weatherStations.flatMap(station -> station.getMeasurements().iterator());
        // Filtering all the temperatures in the range t-1 to t+1.
        JavaRDD<Double> temperature = measurement.filter(m -> m.getTemperature() >= t - 1 && m.getTemperature() <= t + 1)
                .flatMap(m -> Arrays.asList(m.getTemperature()).iterator());
        /* Using map-reduce operation to count the temperature t and storing it in a map in the form of
         {temperature: t, count: 3}. */
        Map<Double, Integer> count = temperature.mapToPair((Double s) -> new Tuple2<>(t, 1))
                .reduceByKey(Integer::sum).collectAsMap();

        // Closing the Spark Context.
        ctx.stop();
        ctx.close();
        // Checking if there is no value in the range of t then return 0 else return the actual count.
        return count.get(t) != null ? count.get(t) : 0;
    }
}
