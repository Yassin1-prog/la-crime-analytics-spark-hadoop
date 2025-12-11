import sys
import os
import argparse

# Add project root to sys.path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType, StructType, StructField, StringType
from pyspark.sql.functions import get_json_object, from_json
from utils.timing import run_and_time


def _load_data(spark, data_paths):
    # Load Income Data
    income_df = spark.read.option("header", "true") \
        .option("delimiter", ";") \
        .csv(data_paths["median_income"])
    
    # Load Census Blocks - Read raw JSON only
    blocks_raw = spark.read \
        .option("multiLine", "false") \
        .text(data_paths["census_blocks"])
    
    # Load Crime Data (2020-Present)
    crime_df = spark.read.option("header", "true").csv(data_paths["crime_data_2020_present"])
    
    return income_df, blocks_raw, crime_df

def query5_df(income_df, blocks_raw, crime_df):
    """
    Implementation of Query 5 using DataFrame API and Sedona Spatial Join.
    """
    
    # ==========================================
    # 1. Process Income Data
    # ==========================================
    income_df = income_df.withColumn(
        "Income", 
        F.regexp_replace(F.col("Estimated Median Income"), "[$,]", "").cast(DoubleType())
    ).filter(
        F.col("Income").isNotNull()
    ).select(
        F.col("Zip Code").alias("ZipCode_Income"),
        F.col("Income")
    )

    # ==========================================
    # 2. Process Census Blocks (GeoJSON)
    # ==========================================
    # Define schema for properties
    properties_schema = StructType([
        StructField("COMM", StringType(), True),
        StructField("ZCTA20", StringType(), True),
        StructField("POP20", LongType(), True)
    ])
    
    # Process line by line (more memory efficient)
    blocks_filtered = blocks_raw \
        .filter(F.col("value").contains("\"type\": \"Feature\"")) \
        .withColumn("properties", from_json(
            get_json_object(F.col("value"), "$.properties"),
            properties_schema
        )) \
        .withColumn("geometry_str", get_json_object(F.col("value"), "$.geometry")) \
        .filter(
            (F.col("properties.COMM").isNotNull()) &
            (F.col("properties.ZCTA20").isNotNull()) &
            (F.col("properties.POP20") > 0) &
            (F.col("geometry_str").isNotNull())
        ) \
        .select(
            F.col("properties.COMM").alias("COMM"),
            F.col("properties.POP20").cast(LongType()).alias("POP20"),
            F.col("properties.ZCTA20").alias("ZCTA20"),
            F.expr("ST_GeomFromGeoJSON(geometry_str)").alias("block_geometry")
        ).filter(
            F.col("block_geometry").isNotNull()
        )

    # ==========================================
    # 3. Calculate Community Income & Population
    # ==========================================
    blocks_income_joined = blocks_filtered.join(
        income_df,
        blocks_filtered.ZCTA20 == income_df.ZipCode_Income,
        "inner"
    )

    community_stats = blocks_income_joined.groupBy("COMM").agg(
        F.sum("POP20").alias("Total_Population"),
        (F.sum(F.col("POP20") * F.col("Income")) / F.sum("POP20")).alias("Avg_Income")
    ).filter(
        (F.col("Total_Population") > 0) & 
        (F.col("Avg_Income").isNotNull())
    )

    # ==========================================
    # 4. Process Crime Data
    # ==========================================
    crime_filtered = crime_df.withColumn(
        "Year", F.year(F.to_timestamp(F.col("DATE OCC"), "yyyy MMM dd hh:mm:ss a"))
    ).withColumn(
        "LAT_clean", F.col("LAT").cast(DoubleType())
    ).withColumn(
        "LON_clean", F.col("LON").cast(DoubleType())
    ).select(
        F.expr("ST_Point(LON_clean, LAT_clean)").alias("crime_geometry")
    ).filter(
        F.col("crime_geometry").isNotNull() 
    )
    
    # ==========================================
    # 5. Spatial Join: Crimes -> Communities
    # ==========================================
    blocks_geom_only = blocks_filtered.select("COMM", "block_geometry")
    
    # Use left join to see what's not matching
    crime_with_comm = crime_filtered.join(
        blocks_geom_only,
        F.expr("ST_Contains(block_geometry, crime_geometry)"),
        "inner"
    )
    
    community_crimes = crime_with_comm.groupBy("COMM").count().withColumnRenamed("count", "Total_Crimes")

    # ==========================================
    # 6. Final Calculation & Correlation
    # ==========================================
    final_df = community_stats.join(community_crimes, "COMM", "inner")
    
    final_df = final_df.withColumn(
        "Crimes_Per_Person",
        (F.col("Total_Crimes") / 2.0) / F.col("Total_Population")
    )

    # Calculate correlations with error handling
    corr_all = final_df.stat.corr("Avg_Income", "Crimes_Per_Person")
      
    top_10_income = final_df.orderBy(F.col("Avg_Income").desc()).limit(10)
    corr_top_10 = top_10_income.stat.corr("Avg_Income", "Crimes_Per_Person")

    bottom_10_income = final_df.orderBy(F.col("Avg_Income").asc()).limit(10)
    corr_bottom_10 = bottom_10_income.stat.corr("Avg_Income", "Crimes_Per_Person")


    print("\n" + "="*60)
    print("QUERY 5 RESULTS: Income vs Crime Rate Correlation")
    print("="*60)
    print(f"Correlation (All Communities): {corr_all:.4f}" if corr_all is not None else "Correlation (All Communities): N/A")
    print(f"Correlation (Top 10 Income):   {corr_top_10:.4f}" if corr_top_10 is not None else "Correlation (Top 10 Income): N/A")
    print(f"Correlation (Bottom 10 Income):{corr_bottom_10:.4f}" if corr_bottom_10 is not None else "Correlation (Bottom 10 Income): N/A")
    print("="*60 + "\n")
    
    return final_df

def main():
    parser = argparse.ArgumentParser(description='Run Query 5: Income vs Crime Rate Correlation')
    parser.add_argument('--executors', type=str, default='2',
                        help='Number of executor instances (default: 2)')
    parser.add_argument('--cores', type=str, default='4',
                        help='Number of cores per executor (default: 4)')
    parser.add_argument('--memory', type=str, default='8g',
                        help='Memory per executor (default: 8g)')
    args = parser.parse_args()
    
    from utils.spark_setup import get_spark_session
    from utils.config import DATA_PATHS

    config_options = {
        'spark.executor.instances': args.executors,
        'spark.executor.cores': args.cores,
        'spark.executor.memory': args.memory,
    }
    
    spark = get_spark_session(app_name="Query5Test", config_options=config_options)
    
    print("\n" + "=" * 60)
    print("Running Query 5: Income vs Crime Rate Correlation")
    print(f"Config: {args.executors} executors, {args.cores} cores, {args.memory} memory")
    print("=" * 60)
    
    income_df, blocks_raw, crime_df = _load_data(spark, DATA_PATHS)
    
    exec_time = run_and_time(
        lambda: query5_df(income_df, blocks_raw, crime_df),
        explain=True
    )
    print(f"\nTotal Execution Time: {exec_time:.4f} seconds")
    
    spark.stop()

if __name__ == "__main__":
    main()