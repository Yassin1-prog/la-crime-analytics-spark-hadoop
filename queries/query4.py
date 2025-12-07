import sys
import os

# Add project root to sys.path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pyspark.sql import functions as F
from pyspark.sql.window import Window
# Import Sedona functions
from sedona.sql.st_functions import ST_Point, ST_DistanceSphere
from utils.timing import run_and_time

def _load_data(spark, data_paths):
    """
    Loads crime data and police stations data.
    """
    # Load Crime Data [cite: 26, 27]
    df_2010_2019 = spark.read.csv(data_paths["crime_data_2010_2019"], header=True, inferSchema=True)
    df_2020_present = spark.read.csv(data_paths["crime_data_2020_present"], header=True, inferSchema=True)
    crime_df = df_2010_2019.unionByName(df_2020_present)
    
    # Load Police Stations [cite: 34, 47]
    stations_df = spark.read.csv(data_paths["police_stations"], header=True, inferSchema=True)
    
    return crime_df, stations_df

def query4_df(crime_df, stations_df):
    """
    Calculates the number of crimes closest to each police station and the average distance.
    Uses Apache Sedona for geospatial calculations.
    """
    
    # 1. Filter Null Island (0,0) from Crime Data 
    # We assume valid coordinates are non-zero.
    valid_crime_df = crime_df.filter(
        (F.col("LAT") != 0) & (F.col("LON") != 0) & 
        F.col("LAT").isNotNull() & F.col("LON").isNotNull()
    )

    # 2. Create Geometries 
    # Create points for Crime Data (Longitude, Latitude)
    # Note: Sedona ST_Point takes (x, y) -> (Longitude, Latitude)
    crime_geo = valid_crime_df.withColumn(
        "crime_point", 
        ST_Point(F.col("LON").cast("double"), F.col("LAT").cast("double"))
    ).select("DR_NO", "crime_point") # Keep ID and geometry to reduce shuffle size

    # Create points for Police Stations
    # Assuming standard columns 'x', 'y' or 'long', 'lat' in stations csv. 
    # Based on standard LA datasets, police stations often have 'x' and 'y'.
    # We select DIVISION as the identifier.
    stations_geo = stations_df.withColumn(
        "station_point",
        ST_Point(F.col("x").cast("double"), F.col("y").cast("double"))
    ).select(F.col("DIVISION"), F.col("station_point"))

    # 3. Find Closest Station
    # Strategy: 
    # Since there are only 21 police stations[cite: 47], we can Broadcast Cross Join 
    # the stations to the crime data. This avoids an expensive spatial join shuffle.
    # We then calculate distance and keep the minimum per crime.
    
    joined_df = crime_geo.crossJoin(F.broadcast(stations_geo))
    
    # Calculate Distance (in meters) using ST_DistanceSphere
    # ST_Distance would return degrees (Euclidean), Sphere returns meters on earth surface
    with_distance = joined_df.withColumn(
        "distance_meters", 
        ST_DistanceSphere("crime_point", "station_point")
    )

    # Window to find the nearest station for each crime
    w = Window.partitionBy("DR_NO").orderBy(F.col("distance_meters").asc())
    
    closest_station = with_distance.withColumn("rnk", F.rank().over(w)) \
        .filter(F.col("rnk") == 1) \
        .select("DIVISION", "distance_meters")

    # 4. Aggregation [cite: 75]
    # Group by station, count incidents, average distance
    # Convert meters to kilometers for readability (matching spec example magnitude ~2.0)
    result_df = closest_station.groupBy("DIVISION").agg(
        F.count("*").alias("count"),
        F.avg("distance_meters").alias("avg_dist_meters")
    )

    # 5. Formatting 
    final_df = result_df.select(
        F.col("DIVISION").alias("division"),
        F.format_number(F.col("avg_dist_meters") / 1000, 3).alias("average_distance"), # Convert to km
        F.col("count").alias("#")
    ).orderBy(F.col("#").desc())

    return final_df

def run_query_4(spark, data_paths, explain=False):
    crime_df, stations_df = _load_data(spark, data_paths)
    
    # Pass explain=True to run_and_time to see the join strategy chosen by Spark 
    exec_time = run_and_time(
        lambda: query4_df(crime_df, stations_df), 
        explain=explain
    )
    return exec_time

def main():
    from utils.spark_setup import get_spark_session
    from utils.config import DATA_PATHS

    # Spec requires running with specific resources [cite: 103]
    # Note: Resource config is handled by the submit script or session builder.
    # Here we set default local config for testing.
    config_options = {
        'spark.executor.instances': '2',
        'spark.executor.cores': '1',
        'spark.executor.memory': '2g'
    }
    
    spark = get_spark_session(app_name="Query4Test", config_options=config_options)
    
    print("\n" + "=" * 60)
    print("Running Query 4: Nearest Police Stations")
    print("=" * 60)
    
    # We enable explain to analyze the Join Strategy (BroadcastNestedLoopJoin likely)
    exec_time = run_query_4(spark, DATA_PATHS, explain=True)
    print(f"Execution Time: {exec_time:.4f} seconds")
    
    spark.stop()

if __name__ == "__main__":
    main()