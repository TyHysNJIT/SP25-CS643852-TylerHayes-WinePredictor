package com.hvel;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import static org.apache.spark.sql.functions.col;


public class WinePredictorCLI {
    public static void main(String[] args) {
        if (args.length != 11) {
            System.err.println("WinePredictorCLI Usage: [fixed acidity] [volatile acidity] [citric acid] [residual sugar] [chlorides] [free sulfur dioxide] [total sulfur dioxide] [density] [pH] [sulphates] [alcohol]");
            System.exit(1);
        }

        // We took in a bunch of strings from the user, now we need to convert input feature strings to doubles
        double[] features = new double[11];
        for (int i = 0; i < 11; i++) {
            try {
                features[i] = Double.parseDouble(args[i]);
            } catch (NumberFormatException e) {
                System.err.println("Parameter Error: " + args[i] + " is invalid");
                System.exit(1);
            }
        }

        // Creating a Spark session, don't really need to do anything in Spark itself, but we need it to load the model
        SparkSession spark = SparkSession.builder()
                .appName("Wine Predictor CLI ")
                .master("local[*]")
                .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
                .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
                .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com")
                .getOrCreate();

        
        try {
            // Try to load the model
            LogisticRegressionModel model = LogisticRegressionModel.load("saved-model");
            // Create a DataFrame with the input features
            Vector input = Vectors.dense(features);
            Dataset<Row> df = spark.createDataFrame(
                java.util.Collections.singletonList(new FeatureRecord(input)), FeatureRecord.class);

            // Make & display predictions
            Dataset<Row> predictions = model.transform(df);
            predictions.select(col("prediction").alias("Predicted Quality")).show(false);
        } catch (Exception e) {
            System.err.println("An error was thrown: " + e.getMessage());
            e.printStackTrace();
        } finally {
            // Kill the Spark session gracefully
            spark.stop();
        }
    }


    // Inner Class: FeatureRecord 
    // This is what Spark is going to use to create our dataframe using a single feild 'features' typed as Vector
    public static class FeatureRecord implements java.io.Serializable {
        private Vector features;

        public FeatureRecord() {}

        public FeatureRecord(Vector features) {
            this.features = features;
        }

        public Vector getFeatures() {
            return features;
        }

        public void setFeatures(Vector features) {
            this.features = features;
        }
    }
}
