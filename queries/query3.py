import sys
import os
import argparse

# Add project root to sys.path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pyspark.sql import functions as F
from utils.timing import run_and_time

def _load_crime_df(spark, crime_data_paths):
    df_2010_2019 = spark.read.csv(crime_data_paths["crime_data_2010_2019"], header=True, inferSchema=True)
    df_2020_present = spark.read.csv(crime_data_paths["crime_data_2020_present"], header=True, inferSchema=True)
    crime_df = df_2010_2019.unionByName(df_2020_present)

    # Drop duplicate DR_NO values (keep first occurrence)
    crime_df = crime_df.dropDuplicates(["DR_NO"])

    return crime_df


def _load_mo_codes(spark, mo_codes_path):    
    mo_codes_df = spark.read.text(mo_codes_path).select(
        F.regexp_extract(F.col("value"), r"^(\d{4})", 1).alias("Code"),
        F.regexp_extract(F.col("value"), r"^\d{4}\s+(.+)$", 1).alias("Description")
    ).filter(F.col("Code") != "")
    
    return mo_codes_df


# -----------------------------
# DataFrame implementation
# -----------------------------
def query3_df(crime_df, mo_codes_df, join_strategy=None):
    # Explode Mocodes column (contains space-separated codes)
    crime_with_codes = crime_df.select(
        F.explode(F.split(F.col("Mocodes"), " ")).alias("Code")
    ).filter(F.col("Code") != "")
    
    # Count occurrences of each code
    code_counts = crime_with_codes.groupBy("Code").count()
    
    # Join with MO codes descriptions
    if join_strategy:
        # Use hint to suggest join strategy
        result_df = code_counts.join(
            mo_codes_df.hint(join_strategy),
            "Code",
            "left"
        )
    else:
        # Let Spark optimizer decide
        result_df = code_counts.join(mo_codes_df, "Code", "left")
    
    # Select and order results
    result_df = result_df.select(
        "Code",
        F.coalesce(F.col("Description"), F.lit("Unknown")).alias("Description"),
        F.col("count").alias("Frequency")
    ).orderBy(F.col("Frequency").desc())
    
    return result_df


# -----------------------------
# RDD implementation
# -----------------------------
def query3_rdd(crime_df, mo_codes_df):
    # Extract Mocodes column as RDD
    mocodes_rdd = crime_df.select("Mocodes").rdd
    
    # Explode space-separated codes and filter empty strings
    codes_rdd = mocodes_rdd.flatMap(
        lambda row: row["Mocodes"].split(" ") if row["Mocodes"] else []
    ).filter(lambda code: code.strip() != "")
    
    # Count occurrences: map to (code, 1) and reduce
    code_counts_rdd = codes_rdd.map(lambda code: (code.strip(), 1)).reduceByKey(lambda a, b: a + b)
    
    # Convert MO codes to RDD for join: (code, description)
    mo_codes_rdd = mo_codes_df.rdd.map(lambda row: (row["Code"], row["Description"]))
    
    # Left outer join to get descriptions
    joined_rdd = code_counts_rdd.leftOuterJoin(mo_codes_rdd)
    
    # Format results: (code, description, count)
    result_rdd = joined_rdd.map(
        lambda x: (x[0], x[1][1] if x[1][1] else "Unknown", x[1][0])
    )
    
    # Sort by frequency (descending)
    sorted_rdd = result_rdd.sortBy(lambda x: x[2], ascending=False)
    
    # Convert to DataFrame
    result_df = sorted_rdd.toDF(["Code", "Description", "Frequency"])
    
    return result_df


# -----------------------------
# Main function to run all modes of Query 3
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description='Run Query 3: Crime Analysis by MO Codes')
    parser.add_argument('--mode', type=str, default='df', choices=['df', 'rdd'],
                        help='Execution mode (default: df)')
    parser.add_argument('--join-strategy', type=str, default=None,
                        choices=['BROADCAST', 'MERGE', 'SHUFFLE_HASH', 'SHUFFLE_REPLICATE_NL'],
                        help='Join strategy hint for DataFrame mode (default: optimizer choice)')
    args = parser.parse_args()
    
    from utils.spark_setup import get_spark_session
    from utils.config import DATA_PATHS

    config_options = {
        'spark.executor.instances': '4',
        'spark.executor.cores': '1',
        'spark.executor.memory': '2g'
    }
    spark = get_spark_session(app_name="Query3Test", config_options=config_options)
    
    MODE = args.mode
    
    crime_df = _load_crime_df(spark, DATA_PATHS)
    mo_codes_df = _load_mo_codes(spark, DATA_PATHS["mo_codes"])
    
    if MODE == "df":
        print("\n" + "=" * 80)
        print("Testing Query 3 with DataFrame API")
        print("=" * 80)
        
        strategy_name = args.join_strategy if args.join_strategy else "Optimizer's Choice"
        print(f"\n--- Join Strategy: {strategy_name} ---")
        exec_time = run_and_time(
            lambda: query3_df(crime_df, mo_codes_df, args.join_strategy),
            explain=True,
            join_strategy_name=strategy_name
        )        
        print(f"Execution Time: {exec_time:.4f} seconds\n")
        
    else:
        # RDD mode
        print(f"\nRunning Query 3 in mode: {MODE}")
        exec_time = run_and_time(lambda: query3_rdd(crime_df, mo_codes_df))
        print(f"Execution Time: {exec_time:.4f} seconds\n")
    
    spark.stop()


# to run as a standalone script for the MODE specified
if __name__ == "__main__":
    main()