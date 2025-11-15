# project/src/spark_setup.py
"""
SparkSession setup utility.
"""
from pyspark.sql import SparkSession

def get_spark_session(app_name="AdvancedDBProject", config_options={}):
    """
    Initializes and returns a SparkSession.
    Applies Sedona configuration as required by the project.
    """
    
    # Base configuration for Apache Sedona 1.6.1 
    # Required for geospatial queries (Q4, Q5)
    # Using Spark 3.5+ 
    builder = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.extensions", "org.apache.sedona.sql.SedonaSqlExtensions") \
        .config("spark.jars.packages", "org.apache.sedona:sedona-spark-3.5_2.12:1.6.1")

    # Apply custom configurations passed as arguments
    # This allows setting executor numbers, cores, and memory dynamically
    # e.g., {'spark.executors.instances': '4', 'spark.executor.cores': '1', ...}
    for key, value in config_options.items():
        builder = builder.config(key, value)

    spark = builder.getOrCreate()
    
    # Register Sedona UDFs
    # Although not used in Q1, it's good practice to register them
    # if the session is intended for geospatial queries later.
    try:
        from sedona.register import SedonaRegistrator
        SedonaRegistrator.registerAll(spark)
    except ImportError:
        print("Sedona not found or failed to register. Geospatial queries may fail.")

    return spark