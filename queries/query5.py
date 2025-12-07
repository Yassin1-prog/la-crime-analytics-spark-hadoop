import sys
import os

# Add project root to sys.path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType
from sedona.spark.sql import ST_Point, ST_Contains
from utils.timing import run_and_time

def _load_data(spark, data_paths):
    """
    Loads Crime, Census Blocks, and Income data.
    """
    # 1. Load Crime Data (2020-Present) and filter for 2020-2021
    # "calculate within the 2-year period 2020-2021"
    df_2020_present = spark.read.csv(data_paths["crime_data_2020_present"], header=True, inferSchema=True)
    
    # Extract Year and Filter
    crime_df = df_2020_present.withColumn(
        "year", 
        F.year(F.to_timestamp(F.col("DATE OCC"), "yyyy MMM dd hh:mm:ss a"))
    ).filter(F.col("year").isin([2020, 2021]))

    # 2. Load Census Blocks (GeoJSON)
    # Uses Sedona's GeoJSON reader
    census_blocks_df = spark.read.format("geojson").load(data_paths["census_blocks"])
    
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
    # Ensure necessary columns exist (Geometry, COMM, POP20, ZCTA20)
    # Based on standard LA 2020 Census Block schema:
    # COMM: Community Name
    # POP20: Population
    # ZCTA20: Zip Code
    census_df = census_blocks_df.select(
        F.col("geometry"),
        F.col("properties.COMM"),
        F.col("properties.POP20").cast(IntegerType()).alias("population"),
        F.col("properties.ZCTA20").cast(IntegerType()).alias("geo_zip")
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

    comm_stats_df = blocks_with_income.groupBy("COMM").agg(
        F.sum("population").alias("total_pop"),
        (F.sum(F.col("parsed_income") * F.col("population")) / F.sum("population")).alias("avg_income")
    ).filter(F.col("total_pop") > 0) # Filter out unpopulated areas


    # --- SPATIAL JOIN: CRIMES PER COMMUNITY ---

    # Join Crimes to Census Blocks (Spatial Join) to find which COMM they belong to.
    # We join valid_crime_df points to census_df polygons.
    # Note: We join to the raw census_df geometries first, then group by COMM.
    
    # Optimization: Broadcast the Census Blocks if they fit in memory (LA Blocks ~80k rows, might be large but likely fits).
    # However, Sedona handles spatial joins efficiently.
    
    crime_joined_df = valid_crime_df.alias("c").join(
        census_df.alias("b"),
        F.expr("ST_Contains(b.geometry, c.crime_point)"),
        "inner"
    ).select("c.year", "b.COMM")

    # Group by COMM to get crime counts
    # We are calculating "Annual Mean Ratio".
    # Total Crimes (2020+2021) / 2 = Average Annual Crimes
    
    crime_counts_df = crime_joined_df.groupBy("COMM").agg(
        F.count("*").alias("total_crimes_2y")
    ).withColumn(
        "avg_annual_crimes", F.col("total_crimes_2y") / 2
    )

    # --- FINAL MERGE & RATE CALCULATION ---

    # Join Income Stats and Crime Counts
    final_df = comm_stats_df.join(crime_counts_df, "COMM", "inner")

    # Calculate Crime Rate (Crimes per Person)
    # "annual mean ratio of crimes per person"
    analysis_df = final_df.withColumn(
        "crime_rate",
        F.col("avg_annual_crimes") / F.col("total_pop")
    ).select("COMM", "avg_income", "crime_rate").cache() # Cache for multiple actions (Correlations)

    # --- ANALYSIS ---

    print("\n--- Correlation Analysis (2020-2021) ---")

    # 1. Overall Correlation
    overall_corr = analysis_df.stat.corr("avg_income", "crime_rate")
    print(f"Overall Correlation (Income vs Crime Rate): {overall_corr:.4f}")

    # 2. Top 10 Income Areas
    # "examining only the 10 areas with the highest ... income"
    top_10_df = analysis_df.orderBy(F.col("avg_income").desc()).limit(10)
    # Note: Spark's stat.corr works on the whole dataframe. To correlation on top 10, we must compute on the filtered DF.
    # Since .limit() returns a DataFrame, we can run stat.corr on it? 
    # Actually stat.corr is an action on the DataFrame. 
    # But limit() might prevent reliable stat.corr in some versions without caching/collecting, 
    # but logically it works.
    top_10_corr = top_10_df.stat.corr("avg_income", "crime_rate")
    print(f"Top 10 High Income Areas Correlation:       {top_10_corr:.4f}")

    # 3. Bottom 10 Income Areas
    bottom_10_df = analysis_df.orderBy(F.col("avg_income").asc()).limit(10)
    bottom_10_corr = bottom_10_df.stat.corr("avg_income", "crime_rate")
    print(f"Bottom 10 Low Income Areas Correlation:     {bottom_10_corr:.4f}")
    
    # Return the dataframe to allow explain plan if needed
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

    # Resource Config
    # Example config (one of the requested configurations)
    config_options = {
        'spark.executor.instances': '2',
        'spark.executor.cores': '4',
        'spark.executor.memory': '8g'
    }
    
    spark = get_spark_session(app_name="Query5Test", config_options=config_options)
    
    print("\n" + "=" * 60)
    print("Running Query 5: Income vs Crime Rate Correlation")
    print("=" * 60)
    
    # Run with explain=True to see join strategies
    exec_time = run_query_5(spark, DATA_PATHS, explain=True)
    print(f"Execution Time: {exec_time:.4f} seconds")
    
    spark.stop()

if __name__ == "__main__":
    main()