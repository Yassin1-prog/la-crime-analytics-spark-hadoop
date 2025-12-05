from utils.spark_setup import get_spark_session
from utils.config import DATA_PATHS
from queries.query1 import run_query1


def main():
    # Initialize Spark session
    config_options = {
        'spark.executor.instances': '4',
        'spark.executor.cores': '1',
        'spark.executor.memory': '2g'
    }
    spark = get_spark_session(app_name="LA_Crime_Analytics", config_options=config_options)
    
    # Define execution modes for Query 1
    modes = ["df", "df_udf", "rdd"]
    mode_descriptions = {
        "df": "DataFrame API (Native)",
        "df_udf": "DataFrame API with UDF",
        "rdd": "RDD API"
    }
    
    print("\n" + "=" * 60)
    print("Running Query 1: Age Groups Analysis")
    print("=" * 60)
    
    for mode in modes:
        print(f"\n--- Mode: {mode_descriptions[mode]} ---")
        
        try:
            result_df, execution_time = run_query1(spark, DATA_PATHS, mode=mode)
            
            print(f"Execution Time: {execution_time:.4f} seconds")
            print("\nResults:")
            result_df.show(truncate=False)
            
        except Exception as e:
            print(f"Error running Query 1 in {mode} mode: {e}")
    
    print("\n" + "=" * 60)
    print("Execution Complete")
    print("=" * 60)
    
    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main()