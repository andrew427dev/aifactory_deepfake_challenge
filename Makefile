prepare:
\tpython scripts/prepare_data.py --config configs/data.yaml

train:
\tpython scripts/train.py --config configs/train.yaml

val:
\tpython scripts/validate.py --config configs/train.yaml

infer:
\tpython scripts/infer.py --config configs/infer.yaml

submit:
\tpython scripts/make_submission.py --config configs/infer.yaml --o submission/submission.csv
