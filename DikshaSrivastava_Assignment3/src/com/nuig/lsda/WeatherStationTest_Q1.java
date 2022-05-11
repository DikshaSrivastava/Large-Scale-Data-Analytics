package com.nuig.lsda;

import java.util.ArrayList;
import java.util.List;

/**
 * The type Weather station tests the weather station class.
 *
 * @author Diksha Srivastava
 * @since 1.0
 */
public class WeatherStationTest_Q1 {

    /**
     * The entry point of application.
     *
     * @param args the input arguments
     */
    public static void main(String[] args) {
        // Creating the first weather station object.
        WeatherStation_Q1 weatherStation = new WeatherStation_Q1();
        weatherStation.setCity("City 1");
        Measurement_Q1 measurement = new Measurement_Q1();
        measurement.setTemperature(20.0);
        measurement.setTime(3);
        List<Measurement_Q1> measurements = new ArrayList<>();
        measurements.add(measurement);
        Measurement_Q1 measurement1 = new Measurement_Q1();
        measurement1.setTemperature(11.7);
        measurement1.setTime(7);
        measurements.add(measurement1);
        Measurement_Q1 measurement2 = new Measurement_Q1();
        measurement2.setTemperature(9.2);
        measurement2.setTime(9);
        measurements.add(measurement2);
        Measurement_Q1 measurement3 = new Measurement_Q1();
        measurement3.setTemperature(18.7);
        measurement3.setTime(5);
        measurements.add(measurement3);
        Measurement_Q1 measurement4 = new Measurement_Q1();
        measurement4.setTemperature(19.2);
        measurement4.setTime(4);
        measurements.add(measurement4);
        weatherStation.setMeasurements(measurements);

        // Creating the second weather station object.
        WeatherStation_Q1 weatherStation1 = new WeatherStation_Q1();
        weatherStation1.setCity("City 2");
        Measurement_Q1 measurement5 = new Measurement_Q1();
        measurement5.setTemperature(19.2);
        measurement5.setTime(1);
        List<Measurement_Q1> measurements1 = new ArrayList<>();
        measurements1.add(measurement5);
        Measurement_Q1 measurement6 = new Measurement_Q1();
        measurement6.setTemperature(19.2);
        measurement6.setTime(10);
        measurements1.add(measurement6);
        Measurement_Q1 measurement7 = new Measurement_Q1();
        measurement7.setTemperature(7.2);
        measurement7.setTime(8);
        measurements1.add(measurement7);
        weatherStation1.setMeasurements(measurements1);
        // Creating a list of existing weather stations
        WeatherStation_Q1.stations.add(weatherStation);
        WeatherStation_Q1.stations.add(weatherStation1);

        // Test Case 1: When some temperatures lie in the range.
        System.out.println("Count of Temperature 20 across all stations having " +
                "temperatures [20.0, 11.7, 9.2, 18.7, 19.2, 19.2, 19.2, 7.2 ]: " + WeatherStation_Q1.countTemperatures(20));
        // Test Case 2: When some temperatures lie in the range.
        System.out.println("Count of Temperature 10 across all stations having " +
                "temperatures [20.0, 11.7, 9.2, 18.7, 19.2, 19.2, 19.2, 7.2 ]: " + WeatherStation_Q1.countTemperatures(10));
        // Test Case 3: When none temperatures are in range.
        System.out.println("Count of Temperature 5 across all stations having " +
                "temperatures [20.0, 11.7, 9.2, 18.7, 19.2, 19.2, 19.2, 7.2 ]: " + WeatherStation_Q1.countTemperatures(5));

    }
}
