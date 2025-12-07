import time

def run_and_time(fn):
    start = time.time()
    result = fn()
    result.show(truncate = False)
    end = time.time()
    return end - start
