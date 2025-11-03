import pandas as pd

REQUIRED_COLUMNS = ["filename", "label"]

def validate_submission(path: str) -> None:
    df = pd.read_csv(path)
    # 컬럼 체크
    if list(df.columns) != REQUIRED_COLUMNS:
        raise ValueError(f"Columns must be exactly {REQUIRED_COLUMNS}, got {list(df.columns)}")
    # 자료형 체크
    if not pd.api.types.is_integer_dtype(df["label"]):
        raise TypeError("label column must be integer dtype (0 or 1).")
    # 결측/중복 체크
    if df.isna().any().any():
        raise ValueError("Submission contains NaN/Null.")
    if df["filename"].duplicated().any():
        raise ValueError("Duplicate filenames in submission.")
    # 값 범위
    bad = ~df["label"].isin([0,1])
    if bad.any():
        raise ValueError(f"label must be 0/1 only. Bad rows: {bad.sum()}")
    print("✅ submission schema: OK")
