import os
import sys
import contextlib

@contextlib.contextmanager
def hidden_prints():
    with open(os.devnull, 'w') as fnull:
        # Save file descriptors
        original_stdout_fd = os.dup(1)
        original_stderr_fd = os.dup(2)

        # Redirect file descriptors 1 and 2
        os.dup2(fnull.fileno(), 1)
        os.dup2(fnull.fileno(), 2)

        try:
            yield
        finally:
            # Restore original file descriptors
            os.dup2(original_stdout_fd, 1)
            os.dup2(original_stderr_fd, 2)
            os.close(original_stdout_fd)
            os.close(original_stderr_fd)