import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import (split, explode, col, desc, count, 
                                   broadcast, length, trim)
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# 1. Αρχικοποίηση Spark Session
spark = SparkSession.builder \
    .appName("Query 3 - MO Codes Analysis & Join Strategies") \
    .getOrCreate()

# 2. Paths (S3 URIs από την εκφώνηση)
crime_data_paths = [
    "s3://initial-notebook-data-bucket-dblab-905418150721/project_data/LA_Crime_Data/LA_Crime_Data_2010_2019.csv",
    "s3://initial-notebook-data-bucket-dblab-905418150721/project_data/LA_Crime_Data/LA_Crime_Data_2020_2025.csv"
]
mo_codes_path = "s3://initial-notebook-data-bucket-dblab-905418150721/project_data/MO_codes.txt"

# ==========================================
# Φόρτωση και Προετοιμασία (Common)
# ==========================================

# A. Φόρτωση Crime Data
df_crime = spark.read.csv(crime_data_paths, header=True, inferSchema=True)

# Κρατάμε μόνο τη στήλη Mocodes και φιλτράρουμε τα Nulls
# Σημαντικό: Τα Mocodes είναι string με κενά (π.χ. "1234 5678"). 
# Πρέπει να κάνουμε SPLIT και EXPLODE για να μετρήσουμε κάθε κωδικό ξεχωριστά.
df_crime_codes = df_crime.filter(col("Mocodes").isNotNull()) \
    .select(explode(split(col("Mocodes"), " ")).alias("MO_Code")) \
    .filter(col("MO_Code") != "") # Αφαίρεση τυχόν κενών strings από το split

# Cache το processed crime data γιατί θα χρησιμοποιηθεί πολλές φορές
df_crime_codes.cache()
df_crime_codes.count()

# B. Φόρτωση MO Codes (Custom Parsing για txt)
# Το αρχείο είναι text, ο κωδικός είναι στην αρχή, μετά κενό, μετά περιγραφή
rdd_text = spark.sparkContext.textFile(mo_codes_path)

def parse_mo_line(line):
    # Split μόνο στο πρώτο κενό (1 split)
    parts = line.split(' ', 1)
    if len(parts) == 2:
        return (parts[0].strip(), parts[1].strip())
    return None

# Δημιουργία DataFrame για τα MO Codes
rdd_mo_parsed = rdd_text.map(parse_mo_line).filter(lambda x: x is not None)
df_mo_codes = spark.createDataFrame(rdd_mo_parsed, ["Code", "Description"])

# Cache τον πίνακα αναφοράς
df_mo_codes.cache()
df_mo_codes.count()

print("--- Start Query 3 Execution ---")

# ==========================================
# Μέρος 1: RDD API Implementation
# ==========================================
print("\n--- RDD API Implementation ---")
start_time_rdd = time.time()

# Μετατροπή των DataFrames σε RDDs
# RDD 1: (Code, 1) -> ReduceByKey
rdd_crime_counts = df_crime_codes.rdd \
    .map(lambda row: (row["MO_Code"], 1)) \
    .reduceByKey(lambda a, b: a + b)

# RDD 2: (Code, Description)
rdd_mo_lookup = df_mo_codes.rdd.map(lambda row: (row["Code"], row["Description"]))

# Join: (Code, (Count, Description))
# Προσοχή: Το join στο RDD δεν έχει optimizer, είναι ακριβό αν δεν γίνει σωστά.
rdd_joined = rdd_crime_counts.join(rdd_mo_lookup)

# Sort: Ταξινόμηση κατά Count (descending)
# Η δομή μετά το join είναι: (Code, (Count, Description))
# Θέλουμε sort με βάση το x[1][0] δηλαδή το Count
rdd_result = rdd_joined.sortBy(lambda x: x[1][0], ascending=False)

# Action
results_rdd = rdd_result.take(5) # Παίρνουμε τα top 5 για εμφάνιση
end_time_rdd = time.time()

print(f"RDD Execution Time: {end_time_rdd - start_time_rdd:.4f} sec")
for code, (count_val, desc_val) in results_rdd:
    print(f"Code: {code}, Count: {count_val}, Desc: {desc_val}")


# ==========================================
# Μέρος 2: DataFrame API & Join Strategies
# ==========================================
print("\n--- DataFrame API & Join Strategies Analysis ---")

# Πρώτα υπολογίζουμε τα counts (Small Aggregation)
df_counts = df_crime_codes.groupBy("MO_Code").count()

# Λίστα με τα Hints που ζητάει η άσκηση
strategies = ["BROADCAST", "MERGE", "SHUFFLE_HASH", "SHUFFLE_REPLICATE_NL"]

for strategy in strategies:
    print(f"\nTesting Join Strategy: {strategy}")
    
    # Force strategy using .hint()
    # Συντάσσουμε το join: Counts JOIN MO_Codes
    # Εφαρμόζουμε το hint στον "δεξιό" πίνακα (df_mo_codes) που είναι ο μικρός πίνακας αναφοράς
    df_joined_df = df_counts.join(df_mo_codes.hint(strategy), 
                                  df_counts["MO_Code"] == df_mo_codes["Code"], 
                                  "inner")
    
    # Sort
    final_df = df_joined_df.orderBy(desc("count"))
    
    # Μέτρηση Χρόνου
    start_time_strat = time.time()
    # Κάνουμε μια ενέργεια (noop) για να εξαναγκάσουμε την εκτέλεση
    final_df.write.format("noop").mode("overwrite").save() 
    end_time_strat = time.time()
    
    print(f"Strategy {strategy} Time: {end_time_strat - start_time_strat:.4f} sec")
    
    # Εμφάνιση του Explain Plan (μόνο το physical plan για συντομία)
    # Ζητείται από την εκφώνηση να δείτε ποια στρατηγική επέλεξε τελικά
    print("Physical Plan excerpt:")
    final_df.explain(mode="simple")

spark.stop()