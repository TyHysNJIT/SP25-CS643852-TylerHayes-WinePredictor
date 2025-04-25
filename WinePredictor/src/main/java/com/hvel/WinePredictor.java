package com.hvel;

import java.io.IOException;
import org.apache.spark.sql.*;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;

public class WinePredictor {
    public static void main(String[] args) {

        // Build a Spark Session, used for distributing the data and training the model
        SparkSession spark_session = SparkSession.builder().appName("Wine Quality Predictor").getOrCreate();

        // Providing the input columns for our data in an array
        String[] inputCols = {"fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density","pH", "sulphates", "alcohol"};

        // Building an assembler to combine our features, since Spark wants a single feature vector it calls 'features'
        VectorAssembler assembler = new VectorAssembler();
                assembler.setInputCols(inputCols);
                assembler.setOutputCol("features");

        // Load our training data
        Dataset<Row> train_data = spark_session.read().option("header", true).option("inferSchema", true)
                .format("csv")
                .load("s3a://cf-templates-id2j68amdy3m-us-east-2/CS643852-M3-A4-Datastore/TrainingDataset.csv");

        Dataset<Row> test_data = spark_session.read().option("header", true).option("inferSchema", true)
                .format("csv")
                .load("s3a://cf-templates-id2j68amdy3m-us-east-2/CS643852-M3-A4-Datastore/ValidationDataset.csv");

        // Apply the assembler to both datasets
        Dataset<Row> train_prepped = assembler.transform(train_data).select("features", "quality");
        Dataset<Row> test_prepped = assembler.transform(test_data).select("features", "quality");

        // Initialize a Logistic Regression Model and set our label/feature columns. I'm maxing the iterations at 50 to prevent overfitting.
        LogisticRegression lr = new LogisticRegression()
                .setLabelCol("quality")
                .setFeaturesCol("features")
                .setMaxIter(50);

        // Fit the model to the training data
        LogisticRegressionModel model = lr.fit(train_prepped);

        // Make predictions on our validation (test) dataset
        Dataset<Row> predictions = model.transform(test_prepped);

        // Create an evaluator for our multiclass model, we want to know the F1 based on the assignment
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction")
                .setMetricName("f1");

        // Store the F1 from the evaluator
        double f1 = evaluator.evaluate(predictions);

        // Print the stored F1 score
        System.out.println("F1: " + f1);

        // Try to save the model
        try {
            model.save("s3a://cf-templates-id2j68amdy3m-us-east-2/CS643852-M3-A4-Datastore/saved-model");
            System.out.println("Model Saved");
        } catch (IOException e) {
            System.err.println("Model Failed to Save" + e.getMessage());
        }

        spark_session.stop();
    }
}
