import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from track_pipeline import run_tracker  # noqa: E402


def test_hung_tracker_is_killed_and_retried_then_fails():
    with tempfile.TemporaryDirectory() as d:
        rc, _, err = run_tracker([sys.executable, "-c", "import time; time.sleep(30)"], d, 0.5,
                              os.path.join(d, "out"), tries=2)
        assert rc == -1 and "hung" in err


def test_normal_tracker_passes_through():
    with tempfile.TemporaryDirectory() as d:
        rc, out, err = run_tracker([sys.executable, "-c", "import sys; print('o'); sys.stderr.write('x')"], d, 10,
                                   os.path.join(d, "out"))
        assert rc == 0 and out == "o\n" and err == "x"
