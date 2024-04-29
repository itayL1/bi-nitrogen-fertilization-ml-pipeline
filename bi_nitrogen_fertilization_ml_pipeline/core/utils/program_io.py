import sys
from contextlib import contextmanager
from io import StringIO
from time import sleep


@contextmanager
def suppress_stdout():
    sleep(0.1)
    # sys.stdout.flush()

    devnull = StringIO()
    original_stdout = sys.stdout
    sys.stdout = devnull
    try:
        yield
    finally:
        sys.stdout = original_stdout
