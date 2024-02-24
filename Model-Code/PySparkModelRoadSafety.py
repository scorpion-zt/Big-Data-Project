from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# Initialiser une session Spark
spark = SparkSession.builder.appName("ModelForAppRoadSafety").getOrCreate()

# Charger le fichier CSV depuis Hadoop HDFS
hadoop_path = "hdfs://127.0.0.1:9000/DATA/Input/data.csv"
accidents_data = spark.read.csv(hadoop_path, header=True, inferSchema=True)

# Exploration des Données
accidents_data.printSchema()
accidents_data.show(5)

# Triez le DataFrame par date
accidents_data = accidents_data.orderBy("date")

# Visualisez les conditions météorologiques par rapport à la gravité des accidents
accidents_data.select("weather_condition", "severity").groupBy("weather_condition", "severity").count().show()

# Convertir les données Spark DataFrame en Pandas DataFrame
accidents_pd = accidents_data.toPandas()

# Créer un DataFrame résumé pour la visualisation
summary_data = accidents_pd.groupby(["weather_condition", "severity"]).size().reset_index(name='count')

# Créer un diagramme à barres
plt.figure(figsize=(12, 6))
for weather_condition, group_data in summary_data.groupby('weather_condition'):
    plt.bar(group_data['severity'], group_data['count'], label=weather_condition)

plt.xlabel('Gravité des accidents')
plt.ylabel('Nombre d\'accidents')
plt.title('Distribution de la gravité des accidents en fonction des conditions météorologiques')
plt.legend()
plt.show()

# Créer un scatter plot pour la localisation des accidents
fig = px.scatter_mapbox(accidents_pd, 
                        lat="latitude", 
                        lon="longitude", 
                        color="severity", 
                        size_max=15, 
                        zoom=9, 
                        mapbox_style="open-street-map",
                        title='Localisation des accidents en fonction de la gravité')
fig.show()

# Nettoyage des Données
accidents_data = accidents_data.dropna()
accidents_data = accidents_data.dropDuplicates()

# Modélisation Prédictive (Exemple avec Random Forest)
# Prétraitement des données
indexer_weather = StringIndexer(inputCol="weather_condition", outputCol="weather_condition_index", handleInvalid="keep")
indexer_time_of_day = StringIndexer(inputCol="time_of_day", outputCol="time_of_day_index", handleInvalid="keep")
encoder_time_of_day = OneHotEncoder(inputCol="time_of_day_index", outputCol="time_of_day_encoded")
indexer_severity = StringIndexer(inputCol="severity", outputCol="severity_index")
assembler = VectorAssembler(inputCols=["weather_condition_index", "time_of_day_encoded", "latitude", "longitude"], outputCol="features")
rf = RandomForestClassifier(labelCol="severity_index", featuresCol="features", seed=42)

# Création du pipeline
pipeline = Pipeline(stages=[indexer_weather, indexer_time_of_day, encoder_time_of_day, indexer_severity, assembler, rf])

# Séparation des données d'entraînement et de test
(training_data, test_data) = accidents_data.randomSplit([0.8, 0.2], seed=42)

# Entraînement du modèle
model = pipeline.fit(training_data)

# Prédiction sur l'ensemble des données
predictions = model.transform(accidents_data)

# Évaluation de la précision du modèle
evaluator = MulticlassClassificationEvaluator(labelCol="severity_index", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Précision du modèle : {accuracy:.2%}")

# Exporter le modèle dans un répertoire (équivalent de joblib.dump)
model_path = "hdfs://127.0.0.1:9000/DATA/Model"
model.save(model_path)

# Arrêter la session Spark
spark.stop()