#pip install flask
from flask import Flask, render_template, request
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, StringType


app = Flask(__name__)

# Initialiser une session Spark
spark = SparkSession.builder.appName("RoadSafetyApp").getOrCreate()

# Charger le modèle PySpark
model_path = "hdfs://127.0.0.1:9000/DATA/Model"
model = PipelineModel.load(model_path)

# Route pour la page d'accueil
@app.route('/')
def home():
    return render_template('index.html')

# Route pour effectuer des prédictions
@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer les données de la requête POST (latitude, longitude, etc.)
    data = request.form.to_dict()

    # Définir le schéma avec les types de données appropriés
    schema = StructType([
        StructField("latitude", DoubleType(), True),
        StructField("longitude", DoubleType(), True),
        StructField("weather_condition", StringType(), True),
        StructField("time_of_day", StringType(), True)
    ])

    # Créer un DataFrame Spark à partir des données de la requête en utilisant le schéma défini
    input_data = spark.createDataFrame([(float(data['latitude']), float(data['longitude']), data['weather_condition'], data['time_of_day'])], schema=schema)

    # Utiliser le modèle pour effectuer des prédictions
    predictions = model.transform(input_data)

    # Récupérer la prédiction
    prediction = predictions.select("prediction").first()[0]

    # Formater la prédiction pour l'afficher dans la page HTML
    prediction_text = "High Risk" if prediction == 1 else "Low Risk"

    # Retourner la prédiction à la page HTML
    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)