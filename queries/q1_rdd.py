# project/src/query1_rdd.py
"""
Query 1 implementation using RDD API.
"""
from pyspark.sql import SparkSession

def categorize_age_rdd(age):
    """
    Python function to categorize age into groups for RDD.
    [cite: 62, 63, 65]
    """
    try:
        age = int(age)
        if age < 18:
            return "Children"
        elif 18 <= age <= 24:
            return "Young adults"
        elif 25 <= age <= 64:
            return "Adults"
        elif age > 64:
            return "Seniors"
        return None
    except (TypeError, ValueError):
        return None

def query1_rdd(spark, crime_data_paths):
    """
    Executes Query 1 using the RDD API.
    
    Query: "Rank, in descending order, the age groups of victims 
    in incidents involving any form of 'aggravated assault'." [cite: 52]
    """
    
    # Load both crime datasets [cite: 38]
    # We use the DataFrame reader for efficient CSV parsing 
    # and then convert to RDD.
    df_2010_2019 = spark.read.csv(crime_data_paths[0], header=True, inferSchema=True)
    df_2020_present = spark.read.csv(crime_data_paths[1], header=True, inferSchema=True)
    
    crime_df = df_2010_2019.unionByName(df_2020_present)

    # Select only the columns we need before converting to RDD
    # This optimizes data shuffling.
    # --- Column Assumptions ---
    # 'Crm Cd Desc' and 'Vict Age'
    crime_rdd = crime_df.select("Crm Cd Desc", "Vict Age").rdd

    # 1. Filter RDD
    filtered_rdd = crime_rdd.filter(
        lambda row: row["Crm Cd Desc"] and \
                    "aggravated assault" in row["Crm Cd Desc"].lower() and \
                    row["Vict Age"] and \
                    isinstance(row["Vict Age"], int) and \
                    row["Vict Age"] > 0
    ) # 

    # 2. Map: Apply categorization logic and prepare for counting
    #    (age_group, 1)
    map_rdd = filtered_rdd.map(
        lambda row: (categorize_age_rdd(row["Vict Age"]), 1)
    )

    # 3. Filter out any 'None' keys from categorization
    valid_map_rdd = map_rdd.filter(lambda x: x[0] is not None)

    # 4. Reduce: Sum counts by key (age group)
    reduce_rdd = valid_map_rdd.reduceByKey(lambda a, b: a + b)

    # 5. Sort: Order by count (value) in descending order
    sorted_rdd = reduce_rdd.sortBy(lambda x: x[1], ascending=False)

    # Convert back to DataFrame for clean output
    result_df = sorted_rdd.toDF(["Age Group", "count"])

    return result_df