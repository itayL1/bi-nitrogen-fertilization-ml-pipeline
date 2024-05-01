import sys
from contextlib import contextmanager
from io import StringIO
from time import sleep


@contextmanager
def suppress_stdout():
    # to make sure the current pending content is passed to the console in time
    sleep(0.1)

    if _running_from_jupyter_notebook():
        from IPython.utils import io
        with io.capture_output():
            yield
    else:
        devnull = StringIO()
        original_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = original_stdout


def _running_from_jupyter_notebook() -> bool:
    return 'ipykernel' in sys.modules
