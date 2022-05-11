package com.nuig.lsda.dto;

import java.util.List;

/**
 * The type Input which acts as an input of key, value pair to the map phase.
 *
 * @author Diksha Srivastava
 * @since 1.0
 */
public class Input {

    private String city;
    private List<Double> temperatures;

    /**
     * Instantiates a new Input.
     *
     * @param city        the city
     * @param temperatures the temperatures
     */
    public Input(String city, List<Double> temperatures) {
        this.city = city;
        this.temperatures = temperatures;
    }

    @Override
    public String toString() {
        return "City='" + city + '\'' +
                ", Temperatures=" + temperatures;
    }

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
     * Gets temperatures.
     *
     * @return the temperatures
     */
    public List<Double> getTemperatures() {
        return temperatures;
    }

    /**
     * Sets temperatures.
     *
     * @param temperatures the temperatures
     */
    public void setTemperatures(List<Double> temperatures) {
        this.temperatures = temperatures;
    }
}
