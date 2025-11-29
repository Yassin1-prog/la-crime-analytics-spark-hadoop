import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, row_number, avg, count, desc, expr, broadcast
from pyspark.sql.window import Window

# Import Sedona modules (Απαιτείται βάσει της πηγής 84)
from sedona.register import SedonaRegistrator
from sedona.utils import SedonaKryoRegistrator, KryoSerializer

# 1. Αρχικοποίηση Spark Session με ρυθμίσεις για Sedona
# Σημείωση: Στο AWS περιβάλλον σας, τα jars είναι ήδη loaded.
spark = SparkSession.builder \
    .appName("Query 4 - Police Stations Proximity") \
    .config("spark.serializer", KryoSerializer.getName) \
    .config("spark.kryo.registrator", SedonaKryoRegistrator.getName) \
    .getOrCreate()

# Εγγραφή των συναρτήσεων της Sedona (ST_Point, ST_Distance, κλπ.)
SedonaRegistrator.registerAll(spark)

# 2. Paths
crime_paths = [
    "s3://initial-notebook-data-bucket-dblab-905418150721/project_data/LA_Crime_Data/LA_Crime_Data_2010_2019.csv",
    "s3://initial-notebook-data-bucket-dblab-905418150721/project_data/LA_Crime_Data/LA_Crime_Data_2020_2025.csv"
]
stations_path = "s3://initial-notebook-data-bucket-dblab-905418150721/project_data/LA_Police_Stations.csv"

# 3. Φόρτωση Δεδομένων
df_crime = spark.read.csv(crime_paths, header=True, inferSchema=True)
df_stations = spark.read.csv(stations_path, header=True, inferSchema=True)

# 4. Προεπεξεργασία & Geo-Features

# A. Αστυνομικά Τμήματα
# Υποθέτουμε ονόματα στηλών X/Y ή LONG/LAT. Προσαρμόστε ανάλογα με το CSV (συνήθως είναι x,y στο LA data).
# Δημιουργούμε Geometry στήλη.
df_stations_geo = df_stations.select(
    col("DIVISION"),
    expr("ST_Point(CAST(x AS Decimal(24,20)), CAST(y AS Decimal(24,20)))").alias("station_geom")
)

# B. Εγκλήματα
# Φιλτράρισμα Null Island (Lat!=0, Lon!=0) όπως ζητάει η πηγή 87
# Και δημιουργία Geometry στήλης
df_crime_geo = df_crime.filter((col("LAT") != 0) & (col("LON") != 0)) \
    .select(
        col("DR_NO"), # Unique ID του εγκλήματος
        expr("ST_Point(CAST(LON AS Decimal(24,20)), CAST(LAT AS Decimal(24,20)))").alias("crime_geom")
    )

print("--- Start Query 4 Execution ---")
start_time = time.time()

# 5. Υλοποίηση Cross Join & Distance
# Χρησιμοποιούμε Broadcast στο df_stations_geo επειδή είναι πολύ μικρό (21 εγγραφές).
# Αυτό θα οδηγήσει σε BroadcastNestedLoopJoin που είναι αποδοτικό εδώ.

df_joined = df_crime_geo.crossJoin(broadcast(df_stations_geo))

# Υπολογισμός απόστασης.
# Χρησιμοποιούμε ST_DistanceSphere για απόσταση σε ΜΕΤΡΑ (πιο ακριβές για γεωγραφικές συντεταγμένες)
# Αν θέλετε degrees χρησιμοποιήστε ST_Distance.
# Διαιρούμε με 1000 για χιλιόμετρα (ή 1609.34 για μίλια) ώστε να ταιριάζει με το "2.208" του πίνακα 3.
df_with_dist = df_joined.withColumn(
    "distance_km", 
    expr("ST_DistanceSphere(crime_geom, station_geom) / 1000")
)

# 6. Εύρεση του Πλησιέστερου (Nearest Neighbor)
# Partition by Crime, Order by Distance ASC -> Rank 1
window_spec = Window.partitionBy("DR_NO").orderBy("distance_km")

df_nearest = df_with_dist.withColumn("rank", row_number().over(window_spec)) \
    .filter(col("rank") == 1)

# 7. Aggregation ανά Αστυνομικό Τμήμα
result_df = df_nearest.groupBy("DIVISION").agg(
    avg("distance_km").alias("average_distance"),
    count("*").alias("count")
).orderBy(desc("count"))

# Formatting για να ταιριάζει με τον Πίνακα 3 (στρογγυλοποίηση)
final_output = result_df.select(
    col("DIVISION").alias("division"),
    expr("ROUND(average_distance, 3)").alias("average_distance"),
    col("count").alias("#")
)

# Explain plan για να δούμε τη στρατηγική Join (Source 102)
print("Execution Plan (Check for BroadcastNestedLoopJoin):")
final_output.explain()

# Action
results = final_output.collect()
end_time = time.time()

print(f"\n--- Execution Time: {end_time - start_time:.4f} sec ---")

# Εκτύπωση αποτελεσμάτων
print(f"{'division':<20} {'average_distance':<20} {'#'}")
for row in results:
    print(f"{row['division']:<20} {row['average_distance']:<20} {row['#']}")

spark.stop()