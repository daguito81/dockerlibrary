from datetime import timedelta
import os
import pandas as pd
import json
import logging

from sqlalchemy.engine import create_engine
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load

# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG
# Operators; we need this to operate!
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

# Globals
DB_URL = os.getenv('DB_URL')

# These args will get passed on to each operator
# You can override them on a per-task basis during operator initialization
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
    # 'wait_for_downstream': False,
    # 'dag': dag,
    # 'sla': timedelta(hours=2),
    # 'execution_timeout': timedelta(seconds=300),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'sla_miss_callback': yet_another_function,
    # 'trigger_rule': 'all_success'
}
dag = DAG(
    'sample_dag_3',
    default_args=default_args,
    description='Simple Data ETL Pipeline',
    schedule_interval=None,
)


def get_oltp_data():
    data = load_iris()
    logging.info("Loaded Data into Memory")
    df = pd.DataFrame(data['data'], columns=data['feature_names'])
    df['target'] = data['target']
    df.columns = ['sepal_l', 'sepal_w', 'petal_l', 'petal_w', 'target']
    logging.info("Started writing data to File")
    df.to_csv("iris_oltp.csv", sep=";", index=False)
    logging.info("Finished writing data to File")


def get_master_data():
    my_data = {
        "index": [0, 1, 2],
        "flower": ["setosa", "versicolor", "virginica"]
    }
    logging.info("Loaded Data into Memory")
    df = pd.DataFrame(my_data)
    logging.info("Started writing data to File")
    df.to_csv("iris_master.csv", sep=';', index=False)
    logging.info("Finished writing data to File")


def process_data():
    logging.info("Getting Data from Files")
    df1 = pd.read_csv("iris_oltp.csv", sep=';')
    df2 = pd.read_csv("iris_master.csv", sep=';')

    logging.info("Joining Data")
    df3 = pd.merge(df1, df2, "inner", left_on="target", right_on="index")
    df3.drop('target', axis=1, inplace=True)
    df3.drop('index', axis=1, inplace=True)
    logging.info("Separating Data into Train/Test Set")
    train, test = train_test_split(df3, test_size=0.2, random_state=101)

    train['target'] = train['flower']
    train.drop('flower', axis=1, inplace=True)
    test['target'] = test['flower']
    test.drop('flower', axis=1, inplace=True)

    logging.info("Started writing data to Disk")
    train.to_csv("iris_train.csv", sep=';', index=False)
    test.to_csv("iris_test.csv", sep=';', index=False)
    logging.info("Finished writing data to Disk")



def load_data_db():
    """
    Load Iris Dataset to Databasse
    """

    data = pd.read_csv("iris_train.csv", sep=';')
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        conn.execute("DROP TABLE IF EXISTS iris_data CASCADE")
        logging.info("Started writing data to DB")
        data.to_sql("iris_data", engine, if_exists='replace', index=False)
        logging.info("Finished writing data to DB")


def train_model(*args, **kwargs):
    logging.info(kwargs)
    engine = create_engine(DB_URL)
    df = pd.read_sql("SELECT * FROM iris_data", engine)
    X = df.drop('target', axis=1)
    y = df['target']
    try:
        trees = kwargs['trees']
    except KeyError:
        logging.info("No Keyword trees found, using default")
        trees = args[0]
    except IndexError:
        logging.info("Now positional argument, using default")
        trees = 100

    model = RandomForestClassifier(n_estimators=int(trees))

    model.fit(X, y)

    dump(model, "rfmodel{}.joblib".format(trees))


def load_test_data(**kwargs):
    ti = kwargs['ti']
    df = pd.read_csv("iris_test.csv", sep=';')
    ti.xcom_push(key='test_data', value=json.dumps(df.to_json()))


def test_model(*args, **kwargs):
    try:
        trees = kwargs['trees']
    except KeyError:
        logging.info("No Keyword trees found, using default")
        trees = args[0]
    except IndexError:
        logging.info("Now positional argument, using default")
        trees = 100

    ti = kwargs['ti']
    test_data = ti.xcom_pull(key='test_data')
    model = load("rfmodel{}.joblib".format(trees))
    df = pd.read_json(json.loads(test_data))
    X_test = df.drop('target', axis=1)
    y_test = df['target']

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    logging.info("Model Accuracy: {}".format(acc))
    ti.xcom_push(key="model_accuracy", value=acc)




get_oltp = PythonOperator(
    task_id='get_oltp_data',
    provide_context=False,
    python_callable=get_oltp_data,
    dag=dag,
)

get_master = PythonOperator(
    task_id='get_master_data',
    provide_context=False,
    python_callable=get_master_data,
    dag=dag,
)

process_data = PythonOperator(
    task_id='process_data',
    provide_context=False,
    python_callable=process_data,
    dag=dag,
)

load_db = PythonOperator(
    task_id='load_data_db',
    provide_context=False,
    python_callable=load_data_db,
    dag=dag,
)

load_test = PythonOperator(
    task_id='load_test_data',
    provide_context=True,
    python_callable=load_test_data,
    dag=dag,
)

finish = BashOperator(
    task_id="finish",
    bash_command = "echo ------ FINISHED ------",
    dag=dag,
)


# for trees in [10, 20, 50, 100, 200]:
trees = 100
train_mod = PythonOperator(
    task_id='train_model_' + str(trees),
    provide_context=True,
    python_callable=train_model,
    op_args = [trees],
    # op_kwargs={'trees', trees},
    dag=dag,
    )

test_mod = PythonOperator(
    task_id='test_model_' + str(trees),
    provide_context=True,
    python_callable=test_model,
    op_args = [trees],
    # op_kwargs={'trees', trees},
    dag=dag,
)
process_data >> load_test >> test_mod >> finish
[get_oltp, get_master] >> process_data >> load_db >> [train_mod] >> test_mod
