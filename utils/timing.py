import time

def run_and_time(fn):
    start = time.time()
    result = fn()
    end = time.time()
    return result, end - start
