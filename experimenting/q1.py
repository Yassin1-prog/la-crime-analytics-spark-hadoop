import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, desc, udf, lower, lit
from pyspark.sql.types import StringType, IntegerType

# 1. Αρχικοποίηση Spark Session
spark = SparkSession.builder \
    .appName("Query 1 - Age Groups Analysis") \
    .getOrCreate()

# 2. Φόρτωση Δεδομένων
# Χρησιμοποιούμε τα μονοπάτια που δίνονται στον Πίνακα 1 της εκφώνησης 
file_path_2010_2019 = "s3://initial-notebook-data-bucket-dblab-905418150721/project_data/LA_Crime_Data/LA_Crime_Data_2010_2019.csv"
file_path_2020_present = "s3://initial-notebook-data-bucket-dblab-905418150721/project_data/LA_Crime_Data/LA_Crime_Data_2020_2025.csv"

# Φόρτωση και ένωση των δύο αρχείων σε ένα DataFrame
df = spark.read.csv([file_path_2010_2019, file_path_2020_present], header=True, inferSchema=True)

# Καστάρισμα της ηλικίας σε Integer για σιγουριά (αν δεν το έκανε το inferSchema)
df = df.withColumn("Vict Age", col("Vict Age").cast(IntegerType()))

# Φιλτράρισμα: Κρατάμε μόνο περιστατικά "Aggravated Assault" όπως ορίζει το Tip 1 
# Χρησιμοποιούμε lower() για να είμαστε case-insensitive
filtered_df = df.filter(lower(col("Crm Cd Desc")).contains("aggravated assault"))

# Cache το filtered_df επειδή θα χρησιμοποιηθεί και στις 3 μετρήσεις
# για να είναι δίκαιη η σύγκριση της λογικής (και όχι του I/O)
filtered_df.cache()
filtered_df.count() # Trigger action για να γίνει το cache

print("--- Start Query 1 Execution ---")

# ==========================================
# Υλοποίηση 1: Native DataFrame API (Catalyst Optimizer)
# ==========================================
start_time_native = time.time()

# Ορισμός των ηλικιακών ομάδων σύμφωνα με τα citations 
res_native = filtered_df.withColumn("AgeGroup", 
    when(col("Vict Age") < 18, "Children")
    .when((col("Vict Age") >= 18) & (col("Vict Age") <= 24), "Young Adults")
    .when((col("Vict Age") >= 25) & (col("Vict Age") <= 64), "Adults")
    .otherwise("Elderly") # > 64
).groupBy("AgeGroup").count().orderBy(desc("count"))

# Action για να εκτελεστεί το query
native_results = res_native.collect()
end_time_native = time.time()

print(f"\n--- Native DataFrame API Results ({end_time_native - start_time_native:.4f} sec) ---")
for row in native_results:
    print(f"{row['AgeGroup']}: {row['count']}")


# ==========================================
# Υλοποίηση 2: DataFrame API με UDF
# ==========================================
# Ορισμός της Python συνάρτησης
def classify_age(age):
    if age is None: return "Unknown"
    if age < 18: return "Children"
    elif 18 <= age <= 24: return "Young Adults"
    elif 25 <= age <= 64: return "Adults"
    else: return "Elderly"

# Εγγραφή της UDF στο Spark
age_udf = udf(classify_age, StringType())

start_time_udf = time.time()

res_udf = filtered_df.withColumn("AgeGroup", age_udf(col("Vict Age"))) \
                     .groupBy("AgeGroup").count().orderBy(desc("count"))

udf_results = res_udf.collect()
end_time_udf = time.time()

print(f"\n--- DataFrame with UDF Results ({end_time_udf - start_time_udf:.4f} sec) ---")
# Εκτυπώνουμε ενδεικτικά για επιβεβαίωση
for row in udf_results:
    print(f"{row['AgeGroup']}: {row['count']}")


# ==========================================
# Υλοποίηση 3: RDD API
# ==========================================
start_time_rdd = time.time()

# Μετατροπή σε RDD
rdd = filtered_df.rdd

# Map: (Age, 1) -> ReduceByKey -> Sort
# Σημείωση: Το RDD row προσπελαύνεται με το όνομα της στήλης ή index
rdd_result = rdd \
    .map(lambda row: (classify_age(row["Vict Age"]), 1)) \
    .reduceByKey(lambda x, y: x + y) \
    .sortBy(lambda x: x[1], ascending=False)

final_rdd_list = rdd_result.collect()
end_time_rdd = time.time()

print(f"\n--- RDD API Results ({end_time_rdd - start_time_rdd:.4f} sec) ---")
for group, count in final_rdd_list:
    print(f"{group}: {count}")

spark.stop()