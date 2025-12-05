import sys
import os

# Add project root to sys.path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import IntegerType
from utils.timing import run_and_time

def _load_data(spark, data_paths):
    """
    Loads Crime Data and Race/Ethnicity Codes.
    """
    # Load Crime Data (2010-2019 and 2020-Present)
    df_2010_2019 = spark.read.csv(data_paths["crime_data_2010_2019"], header=True, inferSchema=True)
    df_2020_present = spark.read.csv(data_paths["crime_data_2020_present"], header=True, inferSchema=True)
    crime_df = df_2010_2019.unionByName(df_2020_present)
    
    # Load Race Codes
    race_codes_df = spark.read.csv(data_paths["race_codes"], header=True, inferSchema=True)
    
    return crime_df, race_codes_df

# -----------------------------
# DataFrame Implementation
# -----------------------------
def query2_df(crime_df, race_codes_df):
    # 1. Extract Year from DATE OCC
    # Note: Using to_timestamp to handle standard LA data format "MM/dd/yyyy hh:mm:ss a" safely
    crime_with_year = crime_df.withColumn(
        "Year", 
        F.year(F.to_timestamp(F.col("DATE OCC"), "MM/dd/yyyy hh:mm:ss a"))
    )

    # 2. Join with Race Codes to get "Vict Descent Full"
    # Performing an inner join to ensure we only count valid codes
    joined_df = crime_with_year.join(race_codes_df, on="Vict Descent", how="inner")

    # 3. Group by Year and Descent, then count
    grouped_df = joined_df.groupBy("Year", "Vict Descent Full").count()

    # 4. Calculate Total Victims per Year (for percentage calculation)
    window_year = Window.partitionBy("Year")
    with_totals = grouped_df.withColumn("year_total", F.sum("count").over(window_year))

    # 5. Calculate Percentage
    with_percent = with_totals.withColumn(
        "%", 
        F.round((F.col("count") / F.col("year_total")) * 100, 1)
    )

    # 6. Rank to find Top 3
    window_rank = Window.partitionBy("Year").orderBy(F.col("count").desc())
    with_rank = with_percent.withColumn("rn", F.row_number().over(window_rank))

    # 7. Filter Top 3 and Select/Rename columns to match Table 2 spec
    result_df = with_rank.filter(F.col("rn") <= 3) \
        .select(
            F.col("Year").alias("year"),
            F.col("Vict Descent Full").alias("Victim Descent"),
            F.col("count").alias("#"),
            F.col("%")
        ) \
        .orderBy(F.col("year").desc(), F.col("#").desc())

    return result_df

# -----------------------------
# SQL Implementation
# -----------------------------
def query2_sql(spark, crime_df, race_codes_df):
    # Register Temp Views
    crime_df.createOrReplaceTempView("crime_data")
    race_codes_df.createOrReplaceTempView("race_codes")

    query = """
    WITH CrimeWithYear AS (
        SELECT 
            year(to_timestamp(`DATE OCC`, 'MM/dd/yyyy hh:mm:ss a')) as Year,
            `Vict Descent`
        FROM crime_data
    ),
    JoinedCounts AS (
        SELECT 
            c.Year,
            r.`Vict Descent Full`,
            COUNT(*) as count
        FROM CrimeWithYear c
        JOIN race_codes r ON c.`Vict Descent` = r.`Vict Descent`
        WHERE c.Year IS NOT NULL
        GROUP BY 1, 2
    ),
    WithTotals AS (
        SELECT 
            Year,
            `Vict Descent Full`,
            count,
            SUM(count) OVER (PARTITION BY Year) as year_total
        FROM JoinedCounts
    ),
    Ranked AS (
        SELECT 
            Year as year,
            `Vict Descent Full` as `Victim Descent`,
            count as `#`,
            ROUND((count / year_total) * 100, 1) as `%`,
            ROW_NUMBER() OVER (PARTITION BY Year ORDER BY count DESC) as rn
        FROM WithTotals
    )
    SELECT year, `Victim Descent`, `#`, `%`
    FROM Ranked
    WHERE rn <= 3
    ORDER BY year DESC, `#` DESC
    """
    
    return spark.sql(query)

# -----------------------------
# Main function for Query 2
# -----------------------------
def run_query_2(spark, data_paths, mode="df"):
    crime_df, race_codes_df = _load_data(spark, data_paths)

    if mode == "df":
        result_df, exec_time = run_and_time(lambda: query2_df(crime_df, race_codes_df))
    elif mode == "sql":
        result_df, exec_time = run_and_time(lambda: query2_sql(spark, crime_df, race_codes_df))
    else:
        raise ValueError("Invalid mode. Use 'df' or 'sql'.")
    
    return result_df, exec_time

if __name__ == "__main__":
    from utils.spark_setup import get_spark_session
    from utils.config import DATA_PATHS

    # Configuration per specs: 4 executors, 1 core, 2GB RAM [cite: 93]
    config_options = {
        'spark.executor.instances': '4',
        'spark.executor.cores': '1',
        'spark.executor.memory': '2g'
    }
    spark = get_spark_session(app_name="Query2Test", config_options=config_options)
    
    MODE = "df" # Change to "sql" to test SQL implementation

    print(f"\nRunning Query 2 in mode: {MODE}")
    result_df, exec_time = run_query_2(spark, DATA_PATHS, mode=MODE)
    print(f"Execution Time: {exec_time:.4f} seconds")
    result_df.show(n=20, truncate=False)
    
    spark.stop()