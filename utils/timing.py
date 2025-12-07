import time

def run_and_time(fn, explain = False, join_strategy = None):
    start = time.time()
    result = fn()
    result.show(truncate = False)
    end = time.time()

    if explain:
        print("\n" + "=" * 80)
        print(f"Query Execution Plan (Join Strategy: {join_strategy or 'Optimizer Choice'})")
        print("=" * 80)
        result.explain(extended=True)
        print("=" * 80 + "\n")

    return end - start
