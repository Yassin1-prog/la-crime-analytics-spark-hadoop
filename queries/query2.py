import sys
import os

# Add project root to sys.path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from utils.timing import run_and_time

def _load_data(spark, data_paths):
    """
    Loads crime data and race/ethnicity codes.
    """
    # Load Crime Data (2010-2019 and 2020-Present)
    df_2010_2019 = spark.read.csv(data_paths["crime_data_2010_2019"], header=True, inferSchema=True)
    df_2020_present = spark.read.csv(data_paths["crime_data_2020_present"], header=True, inferSchema=True)
    crime_df = df_2010_2019.unionByName(df_2020_present)
    
    # Load Race Codes
    re_codes_df = spark.read.csv(data_paths["race_codes"], header=True, inferSchema=True)
    
    return crime_df, re_codes_df

# -----------------------------
# DataFrame API Implementation
# -----------------------------
def query2_df(crime_df, re_codes_df):
    # 1. Extract Year from DATE OCC
    # Format based on description: "yyyy MMM dd hh:mm:ss a"
    # We transform string to timestamp, then extract year
    crime_with_year = crime_df.withColumn(
        "year", 
        F.year(F.to_timestamp(F.col("DATE OCC"), "yyyy MMM dd hh:mm:ss a"))
    )

    # 2. Join with Race Codes to get Full Description
    # We use a left join to ensure we keep records even if code mapping is missing (though typically we want the description)
    joined_df = crime_with_year.join(
        re_codes_df, 
        crime_with_year["Vict Descent"] == re_codes_df["Vict Descent"], 
        "inner"
    ).select(
        F.col("year"),
        F.col("Vict Descent Full").alias("Victim_Descent")
    )

    # 3. Group by Year and Descent to get counts
    grouped_counts = joined_df.groupBy("year", "Victim_Descent").count()

    # 4. Window functions for Total per Year and Ranking
    window_year = Window.partitionBy("year")
    window_rank = Window.partitionBy("year").orderBy(F.col("count").desc())

    final_df = grouped_counts.withColumn(
        "total_victims_year", 
        F.sum("count").over(window_year)
    ).withColumn(
        "percentage", 
        F.format_number((F.col("count") / F.col("total_victims_year")) * 100, 1)
    ).withColumn(
        "rank", 
        F.rank().over(window_rank)
    )

    # 5. Filter top 3 and format output
    result = final_df.filter(F.col("rank") <= 3)\
        .select(
            F.col("year"),
            F.col("Victim_Descent").alias("Victim Descent"),
            F.col("count").alias("#"),
            F.col("percentage").alias("%")
        )\
        .orderBy(F.col("year").desc(), F.col("#").desc())

    return result

# -----------------------------
# SQL API Implementation
# -----------------------------
def query2_sql(spark, crime_df, re_codes_df):
    # Register Temporary Views
    crime_df.createOrReplaceTempView("crime_data")
    re_codes_df.createOrReplaceTempView("re_codes")

    sql_query = """
    WITH CrimeWithYear AS (
        SELECT 
            year(to_timestamp(`DATE OCC`, 'yyyy MMM dd hh:mm:ss a')) as year,
            `Vict Descent` as v_code
        FROM crime_data
    ),
    JoinedData AS (
        SELECT 
            c.year,
            r.`Vict Descent Full` as victim_desc
        FROM CrimeWithYear c
        JOIN re_codes r ON c.v_code = r.`Vict Descent`
    ),
    GroupedCounts AS (
        SELECT 
            year,
            victim_desc,
            COUNT(*) as victims_count
        FROM JoinedData
        GROUP BY year, victim_desc
    ),
    CalculatedMetrics AS (
        SELECT 
            year,
            victim_desc,
            victims_count,
            SUM(victims_count) OVER (PARTITION BY year) as total_year,
            RANK() OVER (PARTITION BY year ORDER BY victims_count DESC) as rnk
        FROM GroupedCounts
    )
    SELECT 
        year,
        victim_desc as `Victim Descent`,
        victims_count as `#`,
        format_number((victims_count / total_year) * 100, 1) as `%`
    FROM CalculatedMetrics
    WHERE rnk <= 3
    ORDER BY year DESC, victims_count DESC
    """
    
    return spark.sql(sql_query)

# -----------------------------
# Main Runner for Query 2
# -----------------------------
def main():
    from utils.spark_setup import get_spark_session
    from utils.config import DATA_PATHS

    # 4 executors, 1 core, 2GB memory
    config_options = {
        'spark.executor.instances': '4',
        'spark.executor.cores': '1',
        'spark.executor.memory': '2g'
    }
    
    spark = get_spark_session(app_name="Query2Test", config_options=config_options)
    
    # Specify the mode here: "df" or "sql"
    MODE = "df"

    print(f"\nRunning Query 2 in mode: {MODE}")
    
    crime_df, re_codes_df = _load_data(spark, DATA_PATHS)

    if MODE == "df":
        exec_time = run_and_time(lambda: query2_df(crime_df, re_codes_df))
    elif MODE == "sql":
        exec_time = run_and_time(lambda: query2_sql(spark, crime_df, re_codes_df))
    else:
        raise ValueError("Invalid mode. Use 'df' or 'sql'.")
    
    print(f"Execution Time: {exec_time:.4f} seconds")
    
    spark.stop()

if __name__ == "__main__":
    main()