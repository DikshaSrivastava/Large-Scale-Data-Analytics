package com.nuig.lsda;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * The Test class used for testing the Weather Station functionalities.
 *
 * @author Diksha Srivastava
 * @since 1.0
 */
public class Test {
    /**
     * The entry point of application.
     *
     * @param args the input arguments
     */
    public static void main(String[] args) {

        // Creating the first weather station object.
        WeatherStation weatherStation = new WeatherStation();
        weatherStation.setCity("City 1");
        Measurement measurement = new Measurement();
        measurement.setTemperature(20.0);
        measurement.setTime(3);
        List<Measurement> measurements = new ArrayList<>();
        measurements.add(measurement);
        Measurement measurement1 = new Measurement();
        measurement1.setTemperature(11.7);
        measurement1.setTime(7);
        measurements.add(measurement1);
        Measurement measurement2 = new Measurement();
        measurement2.setTemperature(-5.4);
        measurement2.setTime(9);
        measurements.add(measurement2);
        Measurement measurement3 = new Measurement();
        measurement3.setTemperature(18.7);
        measurement3.setTime(5);
        measurements.add(measurement3);
        Measurement measurement4 = new Measurement();
        measurement4.setTemperature(20.9);
        measurement4.setTime(4);
        measurements.add(measurement4);
        weatherStation.setMeasurements(measurements);

        //Creating the second weather station object.
        WeatherStation weatherStation1 = new WeatherStation();
        weatherStation1.setCity("City 2");
        Measurement measurement5 = new Measurement();
        measurement5.setTemperature(8.4);
        measurement5.setTime(1);
        List<Measurement> measurements1 = new ArrayList<>();
        measurements1.add(measurement5);
        Measurement measurement6 = new Measurement();
        measurement6.setTemperature(19.2);
        measurement6.setTime(10);
        measurements1.add(measurement6);
        Measurement measurement7 = new Measurement();
        measurement7.setTemperature(7.2);
        measurement7.setTime(8);
        measurements1.add(measurement7);
        weatherStation1.setMeasurements(measurements1);

        System.out.println("---------Question 1: Finding maximum temperature within a given time range--------------");
        //Test 1 for calculating maximum temperature for City 1 weather station
        System.out.println("Temperatures of "+weatherStation.getCity()+" is "+weatherStation.getMeasurements().stream()
                .map(Measurement::getTemperature).collect(Collectors.toList()));
        System.out.println("Maximum Temperature for "+weatherStation.getCity()+" within 1 and 8 is "+weatherStation.maxTemperature(1,8));

        // Test 2 for calculating maximum temperature for City 2 weather station
        System.out.println("Temperatures of "+weatherStation1.getCity()+" is "+weatherStation1.getMeasurements().stream()
                .map(Measurement::getTemperature).collect(Collectors.toList()));
        System.out.println("Maximum Temperature for "+weatherStation1.getCity()+" within 2 and 10 is "+weatherStation1.maxTemperature(2,10));

        // Test 3 for calculating maximum temperature for City 1 weather station with an invalid value
        System.out.println("Temperatures of "+weatherStation.getCity()+" is "+weatherStation.getMeasurements().stream()
                .map(Measurement::getTemperature).collect(Collectors.toList()));
        System.out.println("Maximum Temperature for "+weatherStation.getCity()+" within 2 and 1 is "+weatherStation.maxTemperature(2,1));

        // Creating a list of existing weather stations
        WeatherStation.stations.add(weatherStation);
        WeatherStation.stations.add(weatherStation1);

        System.out.println("---------Question 2: Finding count of temperature t1, t2 using map-reduce--------------");
        System.out.println("-----Test Case 1------");
        // Test 1 for finding the count of temperature within a given range
        WeatherStation weatherStation2 = new WeatherStation();
        System.out.println("Count of Temperatures after performing Reduce phase: "+weatherStation2.countTemperatures(19.0, 10.8, 2.1));
        System.out.println("-----Test Case 2------");
        // Test 2 for finding the count of temperature within a given range
        System.out.println("Count of Temperatures after performing Reduce phase: "+weatherStation2.countTemperatures(10.0, 15.7, 3.0));
    }
}
