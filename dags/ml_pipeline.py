from airflow.decorators import dag, task
# dag의 역할: airflow에서 task들을 관리하는 단위
# task의 역할: 실제로 실행되는 작업을 정의하는 단위
from airflow.operators.python import PythonOperator
# PythonOperator: Python 함수를 실행하는 Operator
from airflow.operators.empty import EmptyOperator
# EmptyOperator: 아무것도 하지 않는 Operator
from pendulum import datetime
# pendulum: 날짜와 시간을 다루는 라이브러리
from module import train


@dag( # dag 데코레이터
    start_date=datetime(2024, 5, 8),
    schedule="@hourly",
    catchup=False, # catchup: 과거의 task를 실행할지 여부
    doc_md=__doc__, # doc_md: dag의 설명. __doc__: 현재 파일의 주석
    default_args={"owner": "Astro", "retries": 3}, # default_args: dag의 기본 인자. owner: 작성자, retries: 재시도 횟수
    tags=["tag 1", "tag 2", "test"] # tags: dag에 부여할 태그
)
def ml_pipeline():
    start_task = EmptyOperator(task_id="start_task")
    
    train_hospital_model_task = PythonOperator(
        task_id="train_hospital_model_task",
        python_callable=train.train_fn_for_hospital # python_callable: 실행할 Python 함수
    )
    create_hospital_model_task = PythonOperator(
        task_id="create_hospital_model_task",
        python_callable=train.transition_model_stage,
        op_kwargs={"model_name": "hospital_model"} # op_kwargs: Python 함수에 전달할 인자
    )
    transition_hospital_model_task = PythonOperator(
        task_id="transition_hospital_model_task",
        python_callable=train.transition_model_stage,
        op_kwargs={"model_name": "hospital_model"}
    )
    
    end_task = EmptyOperator(task_id="end_task")
    
    start_task >> [train_hospital_model_task >> create_hospital_model_task >> transition_hospital_model_task] >> end_task
    # >>: task 간의 의존성을 정의하는 연산자
    
ml_pipeline() # dag 객체 생성