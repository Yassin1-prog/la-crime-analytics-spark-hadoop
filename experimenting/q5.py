import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import (col, count, sum, avg, expr, year, 
                                   to_timestamp, broadcast, abs as sql_abs)
from pyspark.sql.types import IntegerType, DoubleType

# Import Sedona
from sedona.register import SedonaRegistrator
from sedona.utils import SedonaKryoRegistrator, KryoSerializer

# 1. Setup Spark Session με Sedona
spark = SparkSession.builder \
    .appName("Query 5 - Income vs Crime Correlation") \
    .config("spark.serializer", KryoSerializer.getName) \
    .config("spark.kryo.registrator", SedonaKryoRegistrator.getName) \
    .getOrCreate()

SedonaRegistrator.registerAll(spark)

# 2. Paths (S3 URIs)
crime_path_2020 = "s3://initial-notebook-data-bucket-dblab-905418150721/project_data/LA_Crime_Data/LA_Crime_Data_2020_2025.csv"
census_path = "s3://initial-notebook-data-bucket-dblab-905418150721/project_data/LA_Census_Blocks_2020.geojson"
income_path = "s3://initial-notebook-data-bucket-dblab-905418150721/project_data/LA_income_2021.csv"

# 3. Φόρτωση Δεδομένων

# A. Census Blocks (GeoJSON)
# Το Sedona φορτώνει GeoJSON και βάζει τη γεωμετρία στη στήλη 'geometry'
df_census = spark.read.format("geojson").load(census_path)
# Επιλογή χρήσιμων στηλών: COMM (περιοχή), geometry, και πληθυσμός.
# Υποθέτουμε ότι το πεδίο πληθυσμού λέγεται 'pop' ή 'total_pop'. Προσαρμόστε ανάλογα με το schema.
# Εδώ θα υποθέσω ότι υπάρχει στήλη 'total_pop'. Αν χρειαστεί δείτε το schema με df_census.printSchema()
df_census = df_census.select(col("COMM"), col("geometry").alias("census_geom"), col("total_pop").cast(IntegerType()))

# B. Crime Data (2020-)
df_crime = spark.read.csv(crime_path_2020, header=True, inferSchema=True)
# Φιλτράρισμα για έτη 2020-2021 και Null Island
df_crime = df_crime.withColumn("timestamp", to_timestamp(col("DATE OCC"), "MM/dd/yyyy hh:mm:ss a")) \
                   .withColumn("Year", year(col("timestamp"))) \
                   .filter((col("Year").isin([2020, 2021])) & (col("LAT") != 0) & (col("LON") != 0))

# Δημιουργία Geometry Point για τα εγκλήματα
df_crime = df_crime.select(
    col("DR_NO"), 
    expr("ST_Point(CAST(LON AS Decimal(24,20)), CAST(LAT AS Decimal(24,20)))").alias("crime_geom")
)

# C. Income Data
# Προσοχή: Delimiter is ';'
df_income = spark.read.option("delimiter", ";").csv(income_path, header=True, inferSchema=True)
# Μετονομασία στηλών για ευκολία (Zip Code, Median Income)
# Υποθέτουμε ότι η 1η στήλη είναι Zip και η 2η Income.
old_cols = df_income.columns
df_income = df_income.withColumnRenamed(old_cols[0], "Zip_Code") \
                     .withColumnRenamed(old_cols[1], "Median_Income") \
                     .withColumn("Median_Income", col("Median_Income").cast(DoubleType()))

print("--- Start Query 5 Execution ---")

# ============================================================
# Βήμα 1: Spatial Join (Crime Points -> Census Polygons)
# ============================================================
# Βρίσκουμε σε ποια περιοχή (COMM) ανήκει κάθε έγκλημα
# Το Census dataset μπορεί να είναι μεγάλο για broadcast, αλλά δοκιμάζουμε χωρίς hint αρχικά.
# Χρησιμοποιούμε ST_Contains ή ST_Intersects
df_crime_area = df_crime.alias("c").join(
    df_census.alias("b"),
    expr("ST_Contains(b.census_geom, c.crime_geom)")
).select("c.DR_NO", "b.COMM")

# ============================================================
# Βήμα 2: Aggregation (Stats per Area)
# ============================================================
# Αριθμός εγκλημάτων ανά περιοχή
df_crime_counts = df_crime_area.groupBy("COMM").count().withColumnRenamed("count", "total_crimes")

# Πληθυσμός ανά περιοχή (Αθροίζουμε τα blocks που ανήκουν στο ίδιο COMM)
df_pop_area = df_census.groupBy("COMM").agg(sum("total_pop").alias("area_population"))

# ============================================================
# Βήμα 3: Final Dataset (Join Income, Pop, Crimes)
# ============================================================
# Join όλα μαζί στο 'COMM'.
# Υποθέτουμε ότι το Zip Code του Income ταιριάζει με το COMM.
df_final = df_pop_area.join(df_crime_counts, "COMM", "left") \
                      .join(df_income, df_pop_area["COMM"] == df_income["Zip_Code"], "inner") \
                      .na.fill(0, subset=["total_crimes"]) # Αν δεν υπάρχουν εγκλήματα, βάλε 0

# Υπολογισμός Crime Rate Per Person (Annual Average)
# Rate = (Total Crimes / 2 years) / Population
df_stats = df_final.withColumn(
    "crime_rate_per_capita", 
    (col("total_crimes") / 2) / col("area_population")
).filter(col("area_population") > 0) # Αποφυγή διαίρεσης με το 0

df_stats.cache()

# ============================================================
# Μέρος Α: Συσχέτιση για ΟΛΕΣ τις περιοχές
# ============================================================
corr_all = df_stats.stat.corr("Median_Income", "crime_rate_per_capita")
print(f"Correlation (All Areas): {corr_all}")

# ============================================================
# Μέρος Β: Top 10 & Bottom 10 Income Areas
# ============================================================
# Ταξινόμηση
df_sorted = df_stats.orderBy(col("Median_Income").desc())

# Top 10 (Πλουσιότερες)
top_10 = df_sorted.limit(10)
# Bottom 10 (Φτωχότερες) - Χρησιμοποιούμε tail ή sort asc
bottom_10 = df_stats.orderBy(col("Median_Income").asc()).limit(10)

# Ενώνουμε τα δύο sets
df_extremes = top_10.union(bottom_10)

# Υπολογισμός συσχέτισης στα άκρα
corr_extremes = df_extremes.stat.corr("Median_Income", "crime_rate_per_capita")
print(f"Correlation (Top 10 & Bottom 10 Income Areas): {corr_extremes}")

# ============================================================
# Explain Plan & Comments
# ============================================================
print("\n--- Execution Plan for Spatial Join ---")
df_crime_area.explain()

spark.stop()