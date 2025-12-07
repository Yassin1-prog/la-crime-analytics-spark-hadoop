from utils.spark_setup import get_spark_session
from utils.config import DATA_PATHS
from queries.query1 import run_query_1
from queries.query2 import run_query_2
from queries.query3 import run_query_3
from queries.query4 import run_query_4


def main():
    # Initialize Spark session
    config_options = {
        'spark.executor.instances': '4',
        'spark.executor.cores': '1',
        'spark.executor.memory': '2g'
    }
    spark = get_spark_session(app_name="LA_Crime_Analytics", config_options=config_options)
    
    # ========== Query 1 ==========
    print("\n" + "=" * 60)
    print("Running Query 1: Age Groups Analysis")
    print("=" * 60)
    
    modes_q1 = ["df", "df_udf", "rdd"]
    mode_descriptions_q1 = {
        "df": "DataFrame API (Native)",
        "df_udf": "DataFrame API with UDF",
        "rdd": "RDD API"
    }
    
    for mode in modes_q1:
        print(f"\n--- Mode: {mode_descriptions_q1[mode]} ---")
        
        try:
            exec_time = run_query_1(spark, DATA_PATHS, mode=mode)
            print(f"\nExecution Time: {exec_time:.4f} seconds")
            
        except Exception as e:
            print(f"Error running Query 1 in {mode} mode: {e}")
    
    # ========== Query 2 ==========
    print("\n" + "=" * 60)
    print("Running Query 2: Top 3 Victim Descents by Year")
    print("=" * 60)
    
    modes_q2 = ["df", "sql"]
    mode_descriptions_q2 = {
        "df": "DataFrame API",
        "sql": "Spark SQL"
    }
    
    for mode in modes_q2:
        print(f"\n--- Mode: {mode_descriptions_q2[mode]} ---")
        
        try:
            exec_time = run_query_2(spark, DATA_PATHS, mode=mode) 
            print(f"\nExecution Time: {exec_time:.4f} seconds")
            
        except Exception as e:
            print(f"Error running Query 2 in {mode} mode: {e}")
    
    # ========== Query 3 ==========
    print("\n" + "=" * 60)
    print("Running Query 3: Crime Methods (Mocodes) Frequency")
    print("=" * 60)
    
    modes_q3 = ["df", "rdd"]
    mode_descriptions_q3 = {
        "df": "DataFrame API",
        "rdd": "RDD API"
    }
    
    for mode in modes_q3:
        print(f"\n--- Mode: {mode_descriptions_q3[mode]} ---")
        
        try:
            exec_time = run_query_3(spark, DATA_PATHS, mode=mode)
            print(f"\nExecution Time: {exec_time:.4f} seconds")
            
        except Exception as e:
            print(f"Error running Query 3 in {mode} mode: {e}")
    
    print("\n" + "=" * 60)
    print("Execution Complete")
    print("=" * 60)


    # ========== Query 4 ==========
    print("\n" + "=" * 60)
    print("Running Query 4: Nearest Police Stations Analysis")
    print("=" * 60)
    
    try:
        # explain=True will print the physical plan to help analyze the join strategy 
        exec_time = run_query_4(spark, DATA_PATHS, explain=True)
        print(f"\nExecution Time: {exec_time:.4f} seconds")
        
    except Exception as e:
        print(f"Error running Query 4: {e}")

    print("\n" + "=" * 60)
    print("Execution Complete")
    print("=" * 60)

    # ========== Query 5 ==========
    print("\n" + "=" * 60)
    print("Running Query 5: Income vs Crime Correlation")
    print("=" * 60)
    
    try:
        # Explain=True to analyze join strategies
        exec_time = run_query_5(spark, DATA_PATHS, explain=True)
        print(f"\nExecution Time: {exec_time:.4f} seconds")
        
    except Exception as e:
        print(f"Error running Query 5: {e}")
    
    print("\n" + "=" * 60)
    print("Execution Complete")
    print("=" * 60)
    
    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main()