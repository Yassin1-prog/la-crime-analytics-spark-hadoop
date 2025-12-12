# project/src/utils/spark_setup.py
"""
SparkSession setup utility.
"""
from pyspark.sql import SparkSession

def get_spark_session(app_name="AdvancedDBProject", config_options={}):
    """
    Initializes and returns a SparkSession with Sedona and S3 support.
    """
    
    # We need both Sedona (for Q4/Q5) and Hadoop-AWS (for S3 access)
    # The hadoop-aws version (3.3.2) must match the Hadoop version 
    # typically bundled with Spark 3.5.x on SageMaker.
    packages = [
        "org.apache.sedona:sedona-spark-3.5_2.12:1.6.1",
        "org.apache.hadoop:hadoop-aws:3.3.2"
    ]
    
    builder = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.extensions", "org.apache.sedona.sql.SedonaSqlExtensions") \
        .config("spark.jars.packages", ",".join(packages)) \
        .config("spark.hadoop.fs.s3.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.InstanceProfileCredentialsProvider,org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

    # Apply custom configurations passed as arguments
    for key, value in config_options.items():
        builder = builder.config(key, value)

    spark = builder.getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    
    try:
        from sedona.register import SedonaRegistrator
        SedonaRegistrator.registerAll(spark)
    except ImportError:
        print("Sedona not found or failed to register.")

    return spark