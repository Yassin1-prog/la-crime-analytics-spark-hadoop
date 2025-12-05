import sys
import os

# Add project root to sys.path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pyspark.sql import functions as F
from pyspark.sql import Window
from utils.timing import run_and_time


def _load_crime_df(spark, crime_data_paths):
    df_2010_2019 = spark.read.csv(crime_data_paths["crime_data_2010_2019"], header=True, inferSchema=True)
    df_2020_present = spark.read.csv(crime_data_paths["crime_data_2020_present"], header=True, inferSchema=True)
    crime_df = df_2010_2019.unionByName(df_2020_present)
    return crime_df

def _load_race_codes(spark, race_codes_path):
    race_codes_df = spark.read.csv(race_codes_path, header=True, inferSchema=True)
    return race_codes_df


# -----------------------------
# DataFrame implementation
# -----------------------------
def query2_df(crime_df, race_codes_df):
    # Extract year from date
    crime_with_year = crime_df.withColumn(
        "Year", 
        F.year(F.to_date(F.col("DATE OCC"), "MM/dd/yyyy hh:mm:ss a"))
    )
    
    # Join with race codes to get full descriptions
    crime_with_race = crime_with_year.join(
        race_codes_df,
        crime_with_year["Vict Descent"] == race_codes_df["Vict Descent"],
        "left"
    ).select(
        "Year",
        F.coalesce(F.col("Description"), F.col("Vict Descent")).alias("Victim Descent")
    )
    
    # Count victims per year and descent
    victim_counts = crime_with_race.groupBy("Year", "Victim Descent").count()
    
    # Calculate total victims per year
    total_per_year = crime_with_race.groupBy("Year").count().withColumnRenamed("count", "total_count")
    
    # Join to get percentages
    with_percentage = victim_counts.join(total_per_year, "Year")
    with_percentage = with_percentage.withColumn(
        "%",
        F.round((F.col("count") / F.col("total_count")) * 100, 1)
    ).withColumnRenamed("count", "#")
    
    # Rank by count within each year and get top 3
    window_spec = Window.partitionBy("Year").orderBy(F.col("#").desc())
    ranked = with_percentage.withColumn("rank", F.rank().over(window_spec))
    
    # Filter top 3 and order results
    result_df = ranked.filter(F.col("rank") <= 3).select(
        "Year", "Victim Descent", "#", "%"
    ).orderBy(F.col("Year").desc(), F.col("#").desc())
    
    return result_df


# -----------------------------
# SQL implementation
# -----------------------------
def query2_sql(crime_df, race_codes_df, spark):
    # Register DataFrames as temporary views
    crime_df.createOrReplaceTempView("crime_data")
    race_codes_df.createOrReplaceTempView("race_codes")
    
    sql_query = """
    WITH crime_with_year AS (
        SELECT 
            year(to_date(`DATE OCC`, 'MM/dd/yyyy hh:mm:ss a')) AS Year,
            `Vict Descent`
        FROM crime_data
    ),
    crime_with_race AS (
        SELECT 
            c.Year,
            COALESCE(r.Description, c.`Vict Descent`) AS `Victim Descent`
        FROM crime_with_year c
        LEFT JOIN race_codes r ON c.`Vict Descent` = r.`Vict Descent`
    ),
    victim_counts AS (
        SELECT 
            Year,
            `Victim Descent`,
            COUNT(*) AS `#`
        FROM crime_with_race
        GROUP BY Year, `Victim Descent`
    ),
    total_per_year AS (
        SELECT 
            Year,
            COUNT(*) AS total_count
        FROM crime_with_race
        GROUP BY Year
    ),
    with_percentage AS (
        SELECT 
            v.Year,
            v.`Victim Descent`,
            v.`#`,
            ROUND((v.`#` / t.total_count) * 100, 1) AS `%`
        FROM victim_counts v
        JOIN total_per_year t ON v.Year = t.Year
    ),
    ranked AS (
        SELECT 
            Year,
            `Victim Descent`,
            `#`,
            `%`,
            RANK() OVER (PARTITION BY Year ORDER BY `#` DESC) AS rank
        FROM with_percentage
    )
    SELECT 
        Year,
        `Victim Descent`,
        `#`,
        `%`
    FROM ranked
    WHERE rank <= 3
    ORDER BY Year DESC, `#` DESC
    """
    
    result_df = spark.sql(sql_query)
    return result_df


# -----------------------------
# Main function to run all modes of Query 2
# -----------------------------
def run_query_2(spark, data_paths, mode="df"):
    crime_df = _load_crime_df(spark, data_paths)
    race_codes_df = _load_race_codes(spark, data_paths["race_codes"])

    if mode == "df":
        result_df, exec_time = run_and_time(lambda: query2_df(crime_df, race_codes_df))
    elif mode == "sql":
        result_df, exec_time = run_and_time(lambda: query2_sql(crime_df, race_codes_df, spark))
    else:
        raise ValueError("Invalid mode. Use 'df' or 'sql'.")
    
    return result_df, exec_time


# to run as a standalone script for the MODE specified
if __name__ == "__main__":
    from utils.spark_setup import get_spark_session
    from utils.config import DATA_PATHS

    config_options = {
        'spark.executor.instances': '4',
        'spark.executor.cores': '1',
        'spark.executor.memory': '2g'
    }
    spark = get_spark_session(app_name="Query2Test", config_options=config_options)
    
    # Specify the mode here: "df" or "sql"
    MODE = "df"

    print(f"\nRunning Query 2 in mode: {MODE}")
    result_df, exec_time = run_query_2(spark, DATA_PATHS, mode=MODE)
    print(f"Execution Time: {exec_time:.4f} seconds")
    result_df.show(50, truncate=False)
    
    spark.stop()