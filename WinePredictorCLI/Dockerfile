FROM openjdk:11-jre-slim
WORKDIR /app
COPY target/wine-predictor-cli-1.0-jar-with-dependencies.jar app.jar
COPY saved-model saved-model
ENTRYPOINT ["java", "-jar", "app.jar"]
