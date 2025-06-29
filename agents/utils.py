from io import StringIO
import contextlib

@contextlib.contextmanager
def suppress_tqdm():
    with contextlib.redirect_stderr(StringIO()):
        yield
