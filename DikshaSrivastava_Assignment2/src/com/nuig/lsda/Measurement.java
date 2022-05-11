package com.nuig.lsda;

/**
 * The type Measurement which measures temperature for a particular time.
 *
 * @author Diksha Srivastava
 * @since 1.0
 */
public class Measurement {
    private int time;
    private double temperature;

    /**
     * Gets time.
     *
     * @return the time
     */
    public int getTime() {
        return time;
    }

    /**
     * Sets time.
     *
     * @param time the time
     */
    public void setTime(int time) {
        this.time = time;
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

    @Override
    public String toString() {
        return  "Time=" + time +
                ", Temperature=" + temperature ;
    }
}
