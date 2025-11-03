prepare:
	python scripts/prepare_data.py --config configs/data.yaml

train:
	python scripts/train.py --config configs/train.yaml

val:
	python scripts/validate.py --config configs/train.yaml

infer:
	python scripts/infer.py --config configs/infer.yaml

submit:
	python scripts/make_submission.py --config configs/infer.yaml --o submission/submission.csv
