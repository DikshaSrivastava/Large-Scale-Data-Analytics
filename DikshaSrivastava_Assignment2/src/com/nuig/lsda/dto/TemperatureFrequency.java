package com.nuig.lsda.dto;

/**
 * The type Temperature frequency which stores the number of times a particular temperature occurred.
 *
 * @author Diksha Srivastava
 * @since 1.0
 */
public class TemperatureFrequency {

    private double temperature;
    private int count;

    /**
     * Instantiates a new Temperature frequency.
     *
     * @param temperature the temperature
     * @param count       the count
     */
    public TemperatureFrequency(double temperature, int count) {
        this.temperature = temperature;
        this.count = count;
    }

    @Override
    public String toString() {
        return  "(Temperature=" + temperature +
                ", count=" + count +")";
    }

    /**
     * Gets temperature.
     *
     * @return the temperature
     */
    public double getTemperature() {
        return temperature;
    }

    /**
     * Sets temperature.
     *
     * @param temperature the temperature
     */
    public void setTemperature(double temperature) {
        this.temperature = temperature;
    }

    /**
     * Gets count.
     *
     * @return the count
     */
    public int getCount() {
        return count;
    }

    /**
     * Sets count.
     *
     * @param count the count
     */
    public void setCount(int count) {
        this.count = count;
    }

}
