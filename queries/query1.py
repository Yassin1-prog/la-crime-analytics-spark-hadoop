import sys
import os

# Add project root to sys.path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType
from utils.timing import run_and_time


def _load_crime_df(spark, crime_data_paths):
    df_2010_2019 = spark.read.csv(crime_data_paths["crime_data_2010_2019"], header=True, inferSchema=True)
    df_2020_present = spark.read.csv(crime_data_paths["crime_data_2020_present"], header=True, inferSchema=True)
    crime_df = df_2010_2019.unionByName(df_2020_present)
    crime_df = crime_df.withColumn("Vict Age", F.col("Vict Age").cast(IntegerType()))
    return crime_df

# -----------------------------
# DataFrame implementation
# -----------------------------
def query1_df(crime_df):
    filtered_df = crime_df.filter(
        F.lower(F.col("Crm Cd Desc")).contains("aggravated assault") & (F.col("Vict Age") > 0)
    )
    age_groups_df = filtered_df.withColumn(
        "Age Group",
        F.when(F.col("Vict Age") < 18, "Children")
         .when(F.col("Vict Age").between(18, 24), "Young adults")
         .when(F.col("Vict Age").between(25, 64), "Adults")
         .when(F.col("Vict Age") > 64, "Seniors")
         .otherwise("Unknown")
    )
    result_df = age_groups_df.groupBy("Age Group").count().orderBy(F.col("count").desc())
    return result_df

# -----------------------------
# DataFrame + UDF implementation
# -----------------------------
def _categorize_age(age):
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

_categorize_age_udf = F.udf(_categorize_age, StringType())

def query1_df_udf(crime_df):
    filtered_df = crime_df.filter(
        F.lower(F.col("Crm Cd Desc")).contains("aggravated assault") & (F.col("Vict Age") > 0)
    )
    age_groups_df = filtered_df.withColumn("Age Group", _categorize_age_udf(F.col("Vict Age")))
    result_df = age_groups_df.groupBy("Age Group").count().orderBy(F.col("count").desc())
    return result_df

# -----------------------------
# RDD implementation
# -----------------------------
def _categorize_age_rdd(age):
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

def query1_rdd(crime_df):
    crime_rdd = crime_df.select("Crm Cd Desc", "Vict Age").rdd
    filtered_rdd = crime_rdd.filter(
        lambda row: row["Crm Cd Desc"]
        and "aggravated assault" in row["Crm Cd Desc"].lower()
        and row["Vict Age"] is not None
        and isinstance(row["Vict Age"], int)
        and row["Vict Age"] > 0
    )
    mapped = filtered_rdd.map(lambda row: (_categorize_age_rdd(row["Vict Age"]), 1))
    valid = mapped.filter(lambda x: x[0] is not None)
    reduced = valid.reduceByKey(lambda a, b: a + b)
    sorted_rdd = reduced.sortBy(lambda x: x[1], ascending=False)
    result_df = sorted_rdd.toDF(["Age Group", "count"])
    return result_df

# -----------------------------
# Main function to run selected mode of Query 1
# -----------------------------
def main():
    from utils.spark_setup import get_spark_session
    from utils.config import DATA_PATHS

    config_options = {
        'spark.executor.instances': '4',
        'spark.executor.cores': '1',
        'spark.executor.memory': '2g'
    }
    spark = get_spark_session(app_name="Query1Test", config_options=config_options)
    
    # Specify the mode here: "df", "df_udf", or "rdd"
    MODE = "df"

    print(f"\nRunning Query 1 in mode: {MODE}")
    
    crime_df = _load_crime_df(spark, DATA_PATHS)

    if MODE == "df":
        exec_time = run_and_time(lambda: query1_df(crime_df))
    elif MODE == "df_udf":
        exec_time = run_and_time(lambda: query1_df_udf(crime_df))
    elif MODE == "rdd":
        exec_time = run_and_time(lambda: query1_rdd(crime_df))
    else:
        raise ValueError("Invalid mode. Use 'df', 'df_udf', or 'rdd'.")
    
    print(f"Execution Time: {exec_time:.4f} seconds")
    
    spark.stop()


# to run as a standalone script for the MODE specified
if __name__ == "__main__":
    main()