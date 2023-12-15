import time


def timer(func):
    def wrapper(*args, **kwargs):
        print("Starting")
        print("Running ...")
        start = time.perf_counter()
        val = func(*args, **kwargs)
        end = time.perf_counter()
        print("Done!")
        print("Elapsed time = %.5s (s)" % (end - start))
        return val

    return wrapper
