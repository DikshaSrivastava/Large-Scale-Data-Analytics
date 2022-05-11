package com.nuig.lsda;
import java.util.Arrays;
import java.util.Comparator;
import java.util.logging.Logger;

/**
 * 
 * Class Sort is a generic class with type parameter T used to 
 * sort array of objects in a multithreaded environment having
 * length in the power of 2.
 * 
 * @author Diksha Srivastava
 * @param <T> the generic type
 */
public class SortMultithreading<T> { 

	private final static Logger LOGGER = Logger.getLogger(SortMultithreading.class.getName());
	// The array of objects of type T we want to sort
	T[] array; 
	// A Comparator instance suitable for comparing objects of type T
	Comparator<T> comp; 

	/**
	 * The main method.
	 *
	 * @param args the arguments
	 */
	public static void main(String[] args) {

		// Initialising an array of Strings with 16 unordered elements.
		// Array length must be a power of 2.
		String[] arrayOfStrings = { "Blue", "Yellow", "Almond", "Onyx", "Peach", "Gold", "Red", "Melon", "Lava", "Beige", "Aqua", "Lilac", "Capri", "Orange", "Mauve", "Plum" };

		// Initialising with an array of Double with 16 unordered elements.
		// Array length must be a power of 2.
		Double[] arrayOfDouble = {23.45, 56.23, 67.45, 23.0, 24.78, 13.67, 87.89, 98.0, 76.283, 87.65, 90.65, 87.87, 324.56, 11334.5, 467.78, 4657.78 };

		// A comparator for objects of type String:
		Comparator<String> compString = new Comparator<String>() {
			public int compare(String a, String b) {
				if (a.compareTo(b) > 0)
					return 1;
				else
					return 0;
			}
		};
		SortMultithreading<String> sortStrings = new SortMultithreading<>();
		System.out.println("Original array: " + Arrays.toString(arrayOfStrings));
		// Sorting the array by calling the sort-method
		sortStrings.sort(arrayOfStrings, compString);
		System.out.println("Sorted array: " + Arrays.toString(arrayOfStrings));

		// A comparator for type Double
		Comparator<Double> compDouble = new Comparator<Double>() {
			public int compare(Double a, Double b) {
				if(a.compareTo(b) > 0)
					return 1;
				else
					return 0;
			}
		};

		SortMultithreading<Double> sortDouble = new SortMultithreading<>();
		System.out.println("Original Double array: " + Arrays.toString(arrayOfDouble));
		// Sorting the array by calling the sort-method
		sortDouble.sort(arrayOfDouble, compDouble);
		System.out.println("Sorted Double array: " + Arrays.toString(arrayOfDouble));
	}

	/**
	 * Sort method sorts the input array.
	 *
	 * @param array the generic array
	 * @param comp the comparator instance
	 */
	public void sort(T[] array, Comparator<T> comp) { 
		// Array length must be a power of 2
		this.array = array;
		this.comp = comp;
		sort(0, array.length);
	}

	/**
	 * Sort method sorts the input array using divide/conquer strategy and combining the sorted result. 
	 *
	 * @param low the low
	 * @param n the n
	 */
	private void sort(int low, int n) {
		if (n > 1) {
			int mid = n >> 1;
			// Q3 
			// Sorting in a multithreaded environment.

			//Creating a thread instance and calling the recursive sort method.
			Thread t1 = new Thread(new Runnable() {	
				@Override
				public void run() {
					sort(low, mid);

				}
			});

			//Creating a thread instance and calling the recursive sort method.
			Thread t2 = new Thread(new Runnable() {	
				@Override
				public void run() {
					sort(low + mid, mid);

				}
			});

			// Starting the instances of thread so that run method is called.
			t1.start();
			t2.start();
			try {
				// using join so that t2 thread executes only when t1's thread execution completes
				t1.join();
				// using join so that t2 thread completes it's execution first before any other thread 
				t2.join();
			} catch (InterruptedException e) {
				LOGGER.info("Thread Interrupted: "+ e.getMessage());
			}
			combine(low, n, 1);
		}
	}

	/**
	 * Combine method merges the sorted array.
	 *
	 * @param low the lower index
	 * @param n the length of the array
	 * @param st the st
	 */
	private void combine(int low, int n, int st) {
		int m = st << 1;
		if (m < n) {
			combine(low, n, m);
			combine(low + st, n, m);
			for (int i = low + st; i + st < low + n; i += m)
				compareAndSwap(i, i + st);
		} else
			compareAndSwap(low, low + st);
	}

	/**
	 * Compare and swap.
	 *
	 * @param i the i
	 * @param j the j
	 */
	private void compareAndSwap(int i, int j) {
		if (comp.compare(array[i], array[j]) > 0)
			swap(i, j);
	}

	/**
	 * Swap.
	 *
	 * @param i the i
	 * @param j the j
	 */
	private void swap(int i, int j) {
		T h = array[i];
		array[i] = array[j];
		array[j] = h;
	}
}
