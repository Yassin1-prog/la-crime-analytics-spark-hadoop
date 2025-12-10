import sys
import os

# Add project root to sys.path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType
# Import ST_GeomFromGeoJSON for manual parsing
from sedona.spark.sql import ST_Point, ST_Contains, ST_GeomFromGeoJSON
from utils.timing import run_and_time

def _load_data(spark, data_paths):
    """
    Loads Crime, Census Blocks, and Income data.
    """
    # 1. Load Crime Data (2020-Present) and filter for 2020-2021
    df_2020_present = spark.read.csv(data_paths["crime_data_2020_present"], header=True, inferSchema=True)
    
    crime_df = df_2020_present.withColumn(
        "year", 
        F.year(F.to_timestamp(F.col("DATE OCC"), "yyyy MMM dd hh:mm:ss a"))
    ).filter(F.col("year").isin([2020, 2021]))

    # 2. Load Census Blocks - ROBUST METHOD
    # Instead of .format("geojson"), we read as standard JSON to handle nulls safely.
    # We load the whole file as a JSON object, explode the features, and filter bad geometries.
    
    raw_json_df = spark.read.option("multiLine", "true").json(data_paths["census_blocks"])
    
    # Explode the 'features' array (Standard GeoJSON structure)
    features_df = raw_json_df.select(F.explode(F.col("features")).alias("feature"))
    
    # Extract Properties and Geometry String safely
    # We convert the geometry struct back to a JSON string so Sedona can parse it.
    # Crucially, we FILTER null geometry strings before calling Sedona functions.
    census_blocks_df = features_df.select(
        F.col("feature.properties.COMM").alias("COMM"),
        F.col("feature.properties.POP20").alias("POP20"),
        F.col("feature.properties.ZCTA20").alias("ZCTA20"),
        F.to_json(F.col("feature.geometry")).alias("geo_json_str")
    ).filter(
        F.col("geo_json_str").isNotNull() & (F.length(F.col("geo_json_str")) > 0)
    ).withColumn(
        "geometry", ST_GeomFromGeoJSON(F.col("geo_json_str"))
    ).drop("geo_json_str")

    # 3. Load Income Data (CSV with ';' delimiter)
    # "Median Household Income ... delimitter ';'"
    income_df = spark.read.option("delimiter", ";").csv(data_paths["median_income"], header=True, inferSchema=True)

    return crime_df, census_blocks_df, income_df

def query5_execution(spark, crime_df, census_blocks_df, income_df):
    """
    Calculates correlation between Income and Crime Rate per Area (COMM).
    """
    
    # --- PREPROCESSING ---

    # 1. Clean Crime Data: Filter Null Island and Create Points
    # "filter Null Island (0,0)"
    valid_crime_df = crime_df.filter(
        (F.col("LAT") != 0) & (F.col("LON") != 0) & 
        F.col("LAT").isNotNull() & F.col("LON").isNotNull()
    ).withColumn(
        "crime_point", 
        ST_Point(F.col("LON").cast("double"), F.col("LAT").cast("double"))
    ).select("year", "crime_point") # Optimization: Keep only necessary columns

    # 2. Clean Income Data
    # Parse "$52,806" -> 52806.0
    cleaned_income_df = income_df.withColumn(
        "parsed_income",
        F.regexp_replace(F.col("Estimated Median Income"), "[$,]", "").cast(DoubleType())
    ).select(
        F.col("Zip Code").alias("income_zip"), 
        F.col("parsed_income")
    )

    # 3. Prepare Census Data
    census_df = census_blocks_df.select(
        F.col("geometry"),
        F.col("COMM"),
        F.col("POP20").cast(IntegerType()).alias("population"),
        F.col("ZCTA20").cast(IntegerType()).alias("geo_zip")
    )

    # --- AGGREGATION: INCOME PER COMMUNITY ---
    
    # Join Census Blocks with Income based on Zip Code
    # We calculate the Weighted Average Income for each COMM based on the population of its blocks.
    # Logic: A COMM (Community) is composed of many Blocks. Each Block has a Zip (and thus Income) and Population.
    # Avg_Income_COMM = Sum(Income_Block * Pop_Block) / Sum(Pop_Block)
    
    blocks_with_income = census_df.join(
        cleaned_income_df,
        census_df.geo_zip == cleaned_income_df.income_zip,
        "inner"
    )

    # Weighted Average Income per Community
    comm_stats_df = blocks_with_income.groupBy("COMM").agg(
        F.sum("population").alias("total_pop"),
        (F.sum(F.col("parsed_income") * F.col("population")) / F.sum("population")).alias("avg_income")
    ).filter(F.col("total_pop") > 0)


    # --- SPATIAL JOIN: CRIMES PER COMMUNITY ---

    # Spatial Join: Points (Crimes) inside Polygons (Blocks)
    crime_joined_df = valid_crime_df.alias("c").join(
        census_df.alias("b"),
        F.expr("ST_Contains(b.geometry, c.crime_point)"),
        "inner"
    ).select("c.year", "b.COMM")

    # Count Crimes and Calculate Annual Average
    crime_counts_df = crime_joined_df.groupBy("COMM").agg(
        F.count("*").alias("total_crimes_2y")
    ).withColumn(
        "avg_annual_crimes", F.col("total_crimes_2y") / 2
    )

    # --- FINAL MERGE & RATE CALCULATION ---

    final_df = comm_stats_df.join(crime_counts_df, "COMM", "inner")

    # Calculate Crime Rate (Crimes per Person)
    analysis_df = final_df.withColumn(
        "crime_rate",
        F.col("avg_annual_crimes") / F.col("total_pop")
    ).select("COMM", "avg_income", "crime_rate").cache() 

    # --- ANALYSIS ---

    print("\n--- Correlation Analysis (2020-2021) ---")

    # 1. Overall Correlation
    overall_corr = analysis_df.stat.corr("avg_income", "crime_rate")
    print(f"Overall Correlation (Income vs Crime Rate): {overall_corr:.4f}")

    # 2. Top 10 Income Areas
    # "examining only the 10 areas with the highest ... income"
    top_10_df = analysis_df.orderBy(F.col("avg_income").desc()).limit(10)
    top_10_corr = top_10_df.stat.corr("avg_income", "crime_rate")
    print(f"Top 10 High Income Areas Correlation:       {top_10_corr:.4f}")

    # 3. Bottom 10 Income Areas
    bottom_10_df = analysis_df.orderBy(F.col("avg_income").asc()).limit(10)
    bottom_10_corr = bottom_10_df.stat.corr("avg_income", "crime_rate")
    print(f"Bottom 10 Low Income Areas Correlation:     {bottom_10_corr:.4f}")
    
    return analysis_df

def run_query_5(spark, data_paths, explain=False):
    crime_df, census_blocks_df, income_df = _load_data(spark, data_paths)
    
    exec_time = run_and_time(
        lambda: query5_execution(spark, crime_df, census_blocks_df, income_df),
        explain=explain
    )
    return exec_time

def main():
    from utils.spark_setup import get_spark_session
    from utils.config import DATA_PATHS

    config_options = {
        'spark.executor.instances': '2',
        'spark.executor.cores': '4',
        'spark.executor.memory': '8g'
    }
    
    spark = get_spark_session(app_name="Query5Test", config_options=config_options)
    
    print("\n" + "=" * 60)
    print("Running Query 5: Income vs Crime Rate Correlation")
    print("=" * 60)
    
    try: 
        exec_time = run_query_5(spark, DATA_PATHS, explain=True)
        print(f"Execution Time: {exec_time:.4f} seconds")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    spark.stop()

if __name__ == "__main__":
    main()