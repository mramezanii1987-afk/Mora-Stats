"""Entry point for the frozen sidecar binary.

PyInstaller needs a plain script rather than a module, and freezing the
module directly would run it twice under the ``-m`` machinery on Windows.
"""

import multiprocessing

from mora_engine.sidecar import serve

if __name__ == "__main__":
    # Windows re-executes the binary for every worker process unless this is
    # called first, which for a frozen one-file build means the app appears
    # to start several times.
    multiprocessing.freeze_support()
    serve()
