package com.nuig.lsda;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;

import java.util.Arrays;
import java.util.List;

/**
 * The type Linear SVM which trains with IMDB data and do prediction on the model.
 *
 * @author Diksha Srivastava
 * @since 1.0
 */
public class LinearSVM_Q2 {
    /**
     * The entry point of application.
     *
     * @param args the input arguments
     */
    public static void main(String[] args) {
        // Adds hadoop home environment variable.
        System.setProperty("hadoop.home.dir", "C:/winutils");

        // Setting up the spark configuration with 4 threads.
        SparkConf sparkConf = new SparkConf()
                .setAppName("LinearSVM")
                .setMaster("local[4]")
                .set("spark.executor.memory", "1g");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        // Setting log level to error to show logs only when error comes.
        sc.setLogLevel("ERROR");

        // Reading the IMDB file in java RDD.
        String path = "./src/imdb_labelled.txt";
        JavaRDD<String> data = sc.textFile(path);
        // Using Hashing TF to convert features to numerical form.
        final HashingTF tf = new HashingTF(10000);

        /* Splitting the IMDB file data to features and labels by splitting the sentence using tab(\t).
        After splitting, the first element is features, which can be further split by space and transformed
        using Hashing TF. The second element is label 0 or 1 representing a negative or a positive sentiment.
        Some pre-processing steps done on features like removing special characters and converting all
        characters to lower case. */
        JavaRDD<LabeledPoint> transformedData = data.map(sen -> new LabeledPoint(Double.parseDouble(sen.split("\t")[1]),
                tf.transform(Arrays.asList(sen.split("\t")[0]
                        .replaceAll("[,.!?:;]", "")
                        .toLowerCase()
                        .split(" ")))));

        // Splitting the data into training and test set with 60% data in training set and the rest 40% in test set.
        JavaRDD<LabeledPoint> training = transformedData.sample(false, 0.6, 11L);
        training.cache();
        JavaRDD<LabeledPoint> test = transformedData.subtract(training);

        // Training the Linear SVM model with training data.
        SVMModel model = SVMWithSGD.train(training.rdd(), 1000);

        // Creating a sample data of 10 Movie Reviews and allowing the model to predict on these values.
        List<LabeledPoint> sampleData = test.take(10);
        sampleData.forEach(sample -> System.out.println("Predicted Label: " + model.predict(sample.features()) +
                " Actual Label: " + sample.label()));
        // Clearing threshold so that raw scores i.e., confidence score is used to make predictions on the model.
        model.clearThreshold();

        // Q2(b)
        // Allowing the model to predict on the remaining 40% test data
        JavaRDD<Tuple2<Object, Object>> scoreAndLabels = test.map(p ->
                new Tuple2<>(model.predict(p.features()), p.label()));

        // Calculating the accuracy of model by finding AUROC.
        BinaryClassificationMetrics metrics =
                new BinaryClassificationMetrics(JavaRDD.toRDD(scoreAndLabels));
        System.out.println("Area under ROC for linear SVM: " + metrics.areaUnderROC());

        // Closing the spark context.
        sc.stop();
        sc.close();
    }
}
