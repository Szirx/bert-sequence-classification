install:
	pip install -r requirements.txt

train:
	PYTHONPATH=. python src/train.py configs/config.yaml