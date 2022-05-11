package com.nuig.lsda;

import com.nuig.lsda.dto.Input;
import com.nuig.lsda.dto.TemperatureFrequency;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.OptionalDouble;
import java.util.logging.Logger;
import java.util.stream.Collectors;

import static java.util.stream.Collectors.*;

/**
 * The type Weather station which finds the maximum temperature within a given time range
 * and also count the number of times a temperature occurred.
 *
 * @author Diksha Srivastava
 * @since 1.0
 */
public class WeatherStation {

    private final static Logger LOGGER = Logger.getLogger(WeatherStation.class.getName());
    /**
     * The constant stations which stores all the existing weather stations.
     */
    public static List<WeatherStation> stations = new ArrayList<>();

    private String city;
    private List<Measurement> measurements;

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
    public List<Measurement> getMeasurements() {
        return measurements;
    }

    /**
     * Sets measurements.
     *
     * @param measurements the measurements
     */
    public void setMeasurements(List<Measurement> measurements) {
        this.measurements = measurements;
    }

    /**
     * This method finds the maximum temperature within a given time range.
     *
     * @param startTime the start time
     * @param endTime   the end time
     * @return the maximum temperature
     */
    public double maxTemperature(int startTime, int endTime){
        OptionalDouble maxTemp = measurements.stream()
                .filter(measurement -> measurement.getTime() >= startTime && measurement.getTime() <= endTime)
                .mapToDouble(Measurement::getTemperature)
                .max();
        if(maxTemp.isPresent()){
            return maxTemp.getAsDouble();
        } else{
            //Printing Logger in case invalid values of startTime and endTime is entered.
            LOGGER.info("No values found within the given range");
            return 0.0;
        }

    }

    /**
     * This method finds the frequency of temperature within the range
     * t1-r to t1+r and t2-r to t2+r.
     *
     * @param t1 the t 1 temperature
     * @param t2 the t 2 temperature
     * @param r  the r
     * @return the list of temperature and it's frequency
     */
    public List<TemperatureFrequency> countTemperatures(double t1, double t2, double r){

        /*Creating the Input for Map phase which will be a pair of city, and it's temperatures.
         Example: [City1, [20.0, 11.7, -5.4, 18.7, 20.9], City2, [8.4, 19.2, 7.2]]*/
        List<Input> inputs = stations.stream()
                .map(input -> new Input(input.getCity(),
                        input.getMeasurements().stream()
                                .map(Measurement::getTemperature)
                                .collect(toList()))).collect(Collectors.toList());
        System.out.println("Input Values to Map phase: "+ inputs);

        /*The Map phase creates a key, value pair (t1, frequency) and (t2, frequency),
         in our case frequency is 1. If t1 is 19.0 and t2 is 10.8 then the result generated
         would be [(19.0,1), (19.0,1)...(10.8,1)]*/
        List<TemperatureFrequency> intermediateValue = new ArrayList<>();
        intermediateValue.addAll(inputs.parallelStream()
                .flatMap(temp -> temp.getTemperatures().stream()
                        .filter(y -> y >= t1-r && y <= t1+r)
                        .map(y -> new TemperatureFrequency(t1,1)))
                .collect(Collectors.toList()));
        intermediateValue.addAll(inputs.parallelStream()
                .flatMap(temp -> temp.getTemperatures().stream()
                        .filter(y -> y >= t2-r && y <= t2+r)
                        .map(y -> new TemperatureFrequency(t2,1)))
                .collect(Collectors.toList()));
        System.out.println("Intermediate Values Generated after performing Map phase: "+intermediateValue);

        /*The Shuffle phase groups together by the outKeys in the intermediate values
          generated. It will generate output in form of {19.0, [1,1,1,1], 10.8, [1]}*/
        Map<Double, List<TemperatureFrequency>> shuffle = intermediateValue.stream()
                .collect(groupingBy(TemperatureFrequency::getTemperature));
        System.out.println("Values generated after grouping of values in Shuffle Phase: ");
        shuffle.forEach((k, v) -> System.out.println("Temperature:" +k +",  Count: [" +
                v.stream().map(m -> Integer.toString(m.getCount()))
                        .collect(Collectors.joining(","))+"]"));

       /*Reduce phase will work on the grouped values and count the frequency of each
         temperature. It will generate output as [(19.0, 4), (10.8, 1)]*/
        List<TemperatureFrequency> finalList = new ArrayList<>();
        shuffle.forEach(
                (k,v) -> finalList.add(new TemperatureFrequency(k, v.parallelStream()
                        .reduce(0, (t,count) -> t+count.getCount(), Integer::sum)))
        );
        return finalList;
    }
}
