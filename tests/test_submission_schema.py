from src.submission.schema_check import validate_submission
import pandas as pd
import tempfile, os

def test_submission_schema_ok():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "submission.csv")
        pd.DataFrame({"filename":["a.jpg","b.mp4"], "label":[0,1]}).to_csv(p, index=False)
        validate_submission(p)

def test_submission_schema_bad_label():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "submission.csv")
        pd.DataFrame({"filename":["a.jpg"], "label":[2]}).to_csv(p, index=False)
        try:
            validate_submission(p)
            assert False, "Should raise"
        except Exception:
            assert True
