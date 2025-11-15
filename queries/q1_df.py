# project/src/query1_df.py
"""
Query 1 implementation using DataFrame API (without UDF).
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

def query1_df(spark, crime_data_paths):
    """
    Executes Query 1 using the DataFrame API.
    
    Query: "Rank, in descending order, the age groups of victims 
    in incidents involving any form of 'aggravated assault'." [cite: 52]
    
    Age groups:
    - Children: < 18 [cite: 62]
    - Young adults: 18 – 24 [cite: 63]
    - Adults: 25 – 64 [cite: 65]
    - Seniors: > 64 [cite: 65]
    """
    
    # Load both crime datasets [cite: 38]
    df_2010_2019 = spark.read.csv(crime_data_paths[0], header=True, inferSchema=True)
    df_2020_present = spark.read.csv(crime_data_paths[1], header=True, inferSchema=True)
    
    # Combine the datasets
    crime_df = df_2010_2019.unionByName(df_2020_present)

    # --- Column Assumptions ---
    # We assume 'Crm Cd Desc' is the crime description column
    # We assume 'Vict Age' is the victim age column
    # These are standard names for this dataset (links [cite: 54, 55])
    
    # Cast 'Vict Age' to Integer, handling potential nulls
    crime_df = crime_df.withColumn("Vict Age", col("Vict Age").cast(IntegerType()))

    # 1. Filter for 'aggravated assault' incidents 
    # 2. Filter for valid ages (age > 0, as 0 or null is often placeholder)
    filtered_df = crime_df.filter(
        F.lower(F.col("Crm Cd Desc")).contains("aggravated assault") & (F.col("Vict Age") > 0)
    )

    # 3. Define age groups using `when` expression [cite: 62, 63, 65]
    age_groups_df = filtered_df.withColumn(
        "Age Group",
        F.when(F.col("Vict Age") < 18, "Children")
         .when(F.col("Vict Age").between(18, 24), "Young adults")
         .when(F.col("Vict Age").between(25, 64), "Adults")
         .when(F.col("Vict Age") > 64, "Seniors")
         .otherwise("Unknown") # Should not happen due to filter, but good practice
    )

    # 4. Group by age group, count, and order
    result_df = age_groups_df.groupBy("Age Group") \
                             .count() \
                             .orderBy(F.col("count").desc())

    return result_df