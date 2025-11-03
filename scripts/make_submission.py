import argparse, yaml
from src.submission.schema_check import validate_submission

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/infer.yaml")
    ap.add_argument("--o", "--output", dest="out", default="submission/submission.csv")
    args = ap.parse_args()
    # 단순히 파일 스키마만 검증 (infer.py가 이미 생성했다고 가정)
    validate_submission(args.out)

if __name__ == "__main__":
    main()
