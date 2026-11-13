from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

with DAG(
    dag_id="ml_classification_pipeline",
    start_date=datetime(2026, 11, 13),
    schedule="@weekly",
    catchup=False
) as dag:

    prepare = PythonOperator(
        task_id="data_preparation",
        python_callable=prepare_data
    )

    preprocess = PythonOperator(
        task_id="preprocessing",
        python_callable=preprocess_data
    )

    train = PythonOperator(
        task_id="model_training",
        python_callable=train_model
    )

    evaluate = PythonOperator(
        task_id="model_evaluation",
        python_callable=evaluate_model
    )

    prepare >> preprocess >> train >> evaluate