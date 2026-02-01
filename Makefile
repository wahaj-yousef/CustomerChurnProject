venv:
	source venv/bin/activate
run-eda:
	python src/01_data_clean_eda.py

features:
	python src/02_feature_engineering.py

train:
	python src/03_model_training.py

mlflow_ui:
	python3 -m mlflow ui --backend-store-uri ./mlruns --port 1235


api:
	uvicorn src.04_api_service:app --reload
