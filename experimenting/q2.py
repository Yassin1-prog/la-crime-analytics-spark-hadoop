import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import (col, to_timestamp, year, count, sum, 
                                   row_number, desc, round, broadcast)
from pyspark.sql.window import Window

# 1. Αρχικοποίηση Spark Session
spark = SparkSession.builder \
    .appName("Query 2 - Race Groups Analysis") \
    .getOrCreate()

# 2. Ορισμός μονοπατιών αρχείων (S3 Paths από την εκφώνηση)
crime_data_10_19 = "s3://initial-notebook-data-bucket-dblab-905418150721/project_data/LA_Crime_Data/LA_Crime_Data_2010_2019.csv"
crime_data_20_present = "s3://initial-notebook-data-bucket-dblab-905418150721/project_data/LA_Crime_Data/LA_Crime_Data_2020_2025.csv"
race_codes_path = "s3://initial-notebook-data-bucket-dblab-905418150721/project_data/RE_codes.csv"

# 3. Φόρτωση Δεδομένων
# Φορτώνουμε τα δύο crime datasets μαζί
df_crime = spark.read.csv([crime_data_10_19, crime_data_20_present], header=True, inferSchema=True)

# Φορτώνουμε τους κωδικούς φυλών
# Υποθέτουμε ότι το CSV έχει headers. Αν δεν έχει, ίσως χρειαστεί .toDF("Code", "Description")
df_race_codes = spark.read.csv(race_codes_path, header=True, inferSchema=True)
# Μετονομάζουμε τις στήλες του race codes για να είμαστε σίγουροι (συνήθως είναι Code, Description)
# Ας υποθέσουμε ότι η πρώτη στήλη είναι ο κωδικός και η δεύτερη η περιγραφή
old_columns = df_race_codes.columns
df_race_codes = df_race_codes.withColumnRenamed(old_columns[0], "Race_Code") \
                             .withColumnRenamed(old_columns[1], "Race_Description")

# 4. Προεπεξεργασία (Preprocessing) - Κοινή και για τις δύο μεθόδους
# Μετατροπή της ημερομηνίας και εξαγωγή του έτους
# Format: 01/08/2020 12:00:00 AM (τυπικό format του LA Data)
df_crime = df_crime.withColumn("timestamp", to_timestamp(col("DATE OCC"), "MM/dd/yyyy hh:mm:ss a")) \
                   .withColumn("Year", year(col("timestamp")))

# JOIN: Ενώνουμε τα εγκλήματα με τις περιγραφές των φυλών.
# Χρησιμοποιούμε BROADCAST join επειδή το αρχείο κωδικών είναι πολύ μικρό ("Μικρό σύνολο δεδομένων" [cite: 48]).
# Αυτό βελτιώνει δραματικά την απόδοση αποφεύγοντας το shuffle του μεγάλου πίνακα.
df_joined = df_crime.join(broadcast(df_race_codes), 
                          df_crime["Vict Descent"] == df_race_codes["Race_Code"], 
                          "left")

# Φιλτράρουμε πιθανές εγγραφές χωρίς Έτος ή χωρίς Φυλή (προαιρετικό, αλλά καλή πρακτική)
df_base = df_joined.filter(col("Year").isNotNull() & col("Race_Description").isNotNull()) \
                   .select("Year", "Race_Description")

# Cache τα δεδομένα βάσης για να είναι δίκαιη η σύγκριση των αλγορίθμων grouping/window
df_base.cache()
df_base.count() # Trigger action

print("--- Start Query 2 Execution ---")

# ============================================================
# Μέθοδος 1: DataFrame API (Native)
# ============================================================
start_time_df = time.time()

# Βήμα 1: Group by Year και Race για να βρούμε το πλήθος θυμάτων
df_grouped = df_base.groupBy("Year", "Race_Description").agg(count("*").alias("count"))

# Βήμα 2: Ορισμός παραθύρων (Windows)
# Window για κατάταξη (Rank) ανά έτος φθίνουσα
window_rank = Window.partitionBy("Year").orderBy(desc("count"))
# Window για υπολογισμό συνολικών θυμάτων ανά έτος (για το ποσοστό)
window_total = Window.partitionBy("Year")

# Βήμα 3: Εφαρμογή Window functions
df_ranked = df_grouped.withColumn("rank", row_number().over(window_rank)) \
                      .withColumn("total_year", sum("count").over(window_total))

# Βήμα 4: Φιλτράρισμα Top 3, Υπολογισμός Ποσοστού και Formatting
df_result_api = df_ranked.filter(col("rank") <= 3) \
    .withColumn("percentage", round((col("count") / col("total_year")) * 100, 1)) \
    .select(
        col("Year").alias("year"),
        col("Race_Description").alias("Victim Descent"),
        col("count").alias("#"),
        col("percentage").alias("%")
    ) \
    .orderBy(desc("year"), desc("#"))

# Action
results_api = df_result_api.collect()
end_time_df = time.time()

print(f"\n--- DataFrame API Results ({end_time_df - start_time_df:.4f} sec) ---")
# Εμφάνιση των πρώτων αποτελεσμάτων όπως στον Πίνακα 2
df_result_api.show(10, truncate=False)


# ============================================================
# Μέθοδος 2: SQL API
# ============================================================
start_time_sql = time.time()

# Καταχωρούμε το clean DataFrame ως προσωρινό View
df_base.createOrReplaceTempView("crime_data_view")

# Γράφουμε το SQL Query χρησιμοποιώντας CTEs (Common Table Expressions)
# για καθαρό κώδικα που μιμείται τη λογική των Windows
sql_query = """
WITH AggregatedData AS (
    -- Βήμα 1: Grouping
    SELECT 
        Year, 
        Race_Description, 
        COUNT(*) as victims_count
    FROM crime_data_view
    GROUP BY Year, Race_Description
),
RankedData AS (
    -- Βήμα 2: Ranking & Totals
    SELECT 
        Year,
        Race_Description,
        victims_count,
        ROW_NUMBER() OVER (PARTITION BY Year ORDER BY victims_count DESC) as rnk,
        SUM(victims_count) OVER (PARTITION BY Year) as total_year
    FROM AggregatedData
)
-- Βήμα 3: Filtering & Formatting
SELECT 
    Year as year,
    Race_Description as `Victim Descent`,
    victims_count as `#`,
    ROUND((victims_count / total_year) * 100, 1) as `%`
FROM RankedData
WHERE rnk <= 3
ORDER BY year DESC, `#` DESC
"""

df_result_sql = spark.sql(sql_query)

# Action
results_sql = df_result_sql.collect()
end_time_sql = time.time()

print(f"\n--- SQL API Results ({end_time_sql - start_time_sql:.4f} sec) ---")
df_result_sql.show(10, truncate=False)

spark.stop()