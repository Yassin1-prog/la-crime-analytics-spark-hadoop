# project/src/query1_df_udf.py
"""
Query 1 implementation using DataFrame API with a UDF.
[cite: 89]
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType

def categorize_age(age):
    """
    Python function to categorize age into groups.
    [cite: 62, 63, 65]
    """
    if age is None:
        return None
    if age < 18:
        return "Children"
    elif 18 <= age <= 24:
        return "Young adults"
    elif 25 <= age <= 64:
        return "Adults"
    elif age > 64:
        return "Seniors"
    return None

# Register the function as a UDF
categorize_age_udf = F.udf(categorize_age, StringType())

def query1_df_udf(spark, crime_data_paths):
    """
    Executes Query 1 using the DataFrame API with a UDF.
    
    Query: "Rank, in descending order, the age groups of victims 
    in incidents involving any form of 'aggravated assault'."
    """
    
    # Load both crime datasets [cite: 38]
    df_2010_2019 = spark.read.csv(crime_data_paths[0], header=True, inferSchema=True)
    df_2020_present = spark.read.csv(crime_data_paths[1], header=True, inferSchema=True)
    
    # Combine the datasets
    crime_df = df_2010_2019.unionByName(df_2020_present)
    
    # Cast 'Vict Age' to Integer
    crime_df = crime_df.withColumn("Vict Age", col("Vict Age").cast(IntegerType()))

    # 1. Filter for 'aggravated assault' incidents 
    # 2. Filter for valid ages (age > 0)
    filtered_df = crime_df.filter(
        F.lower(F.col("Crm Cd Desc")).contains("aggravated assault") & (F.col("Vict Age") > 0)
    )

    # 3. Apply UDF to create age group column
    age_groups_df = filtered_df.withColumn(
        "Age Group", categorize_age_udf(F.col("Vict Age"))
    )

    # 4. Group by age group, count, and order
    result_df = age_groups_df.groupBy("Age Group") \
                             .count() \
                             .orderBy(F.col("count").desc())

    return result_df