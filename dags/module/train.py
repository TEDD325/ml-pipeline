from airflow.providers.postgres.hooks.postgres import PostgresHook
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris
import optuna
from optuna.storages import RDBStorage
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
import logging
        

def train_fn_for_hospital(**context): 
    # context: Airflow의 task 간 데이터를 전달하는 객체
    # context를 통해 다른 task의 상태 확인/데이터를 받아오거나 전달할 수 있음
    # context를 통해 task 간의 의존성을 설정할 수 있음
    
    mlflow.set_experiment("hospital_model")
    # mlflow의 역할: 모델의 메타데이터를 저장하고, 모델의 성능을 추적하는 역할
    # mlflow.set_experiment: 모델의 메타데이터를 저장할 실험을 설정
    
    hook = PostgresHook(postgres_conn_id="postgres_default")
    # PostgresHook: Airflow에서 Postgres에 연결하기 위한 Hook
    # postgres_conn_id: PostgresHook을 생성할 때 사용할 Postgres 연결 ID
    conn = hook.get_conn() # PostgresHook을 통해 Postgres에 연결
    stmt = """
        SELECT * FROM hospital_train
    """ # SQL문 작성
    data = pd.read_sql(stmt, conn) # Postgres에서 데이터를 불러와 DataFrame으로 저장
    label = data["OC"]
    
    data.drop(columns = ["OC", "inst_id", "openDate"], inplace=True)
    # 전처리 -불필요한 열 제거
    
    x_train, x_valid, y_train, y_valid = train_test_split(
        data, label, test_size=0.3, shuffle=True, stratify=label
    ) # 데이터를 학습 데이터와 검증 데이터로 나눔
    # train_test_split: 데이터를 학습 데이터와 검증 데이터로 나누는 함수
    # test_size: 검증 데이터의 비율
    # shuffle: 데이터를 섞을지 여부
    # stratify: 데이터의 클래스 비율을 유지할지 여부
    # 여기서 k_fold를 쓰지 않는 이유: 데이터의 양이 적기 때문에 k_fold를 사용할 경우 데이터의 양이 적어짐
    
    x_train = x_train.reset_index(drop=True)
    x_valid = x_valid.reset_index(drop=True)
    
    cat_columns = data.select_dtypes(include="object").columns
    num_columns = data.select_dtypes(exclude="object").columns
    # 데이터 타입이 object인 열과 그렇지 않은 열을 구분
    
    print("Categorical columns: ", cat_columns)
    print("Numerical columns: ", num_columns)
    
    preprocessor = ColumnTransformer( # ColumnTransformer: 여러 전처리 단계를 하나로 묶어주는 클래스
        transformers=[ # 전처리 단계를 정의
            ("impute", # 전처리 단계의 이름
             IterativeImputer(), # 전처리 단계. IterativeImputer: 결측치를 예측하여 채워주는 클래스
             num_columns), # 전처리 단계를 적용할 열
            ("scaler", 
             StandardScaler(), # StandardScaler: 데이터를 표준화하는 클래스
             num_columns),
            ("encoding", 
             OneHotEncoder(handle_unknown="ignore", sparse_output=False), # OneHotEncoder: 범주형 데이터를 원-핫 인코딩하는 클래스
                            # handle_unknown: 알 수 없는 범주를 무시할지 여부
                            # sparse_output: 희소 행렬을 반환할지 여부
             cat_columns)
        ]
    )
    
    le = LabelEncoder() # LabelEncoder: 범주형 데이터를 정수로 인코딩하는 클래스
    
    y_train = le.fit_transform(y_train) # y_train을 정수로 인코딩
    y_valid = le.transform(y_valid) # y_valid를 정수로 인코딩
    # y_train을 학습할 때는 fit_transform을 사용하고, y_valid를 변환할 때는 transform을 사용
    # 그 이유는 y_train을 학습할 때는 모든 범주를 학습해야 하지만, y_valid를 변환할 때는 학습된 범주를 사용해야 하기 때문
    
    x_train = preprocessor.fit_transform(x_train) 
    x_valid = preprocessor.transform(x_valid)
    
    def objective(trial): # objective 함수: 하이퍼파라미터 최적화를 위한 목적 함수
        n_estimators = trial.suggest_int("n_estimators", 2, 100) # n_estimators: 랜덤 포레스트의 트리 개수
        max_depth = trial.suggest_int("max_depth", 1, 32) # max_depth: 트리의 최대 깊이
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth) # RandomForestClassifier: 랜덤 포레스트 모델
        model.fit(x_train, y_train) # 모델 학습
        return f1_score(y_valid, model.predict(x_valid)) # f1_score: 모델의 성능을 평가하는 지표
        # f1_score: 정밀도와 재현율의 조화평균을 계산하는 지표
    # objective 함수는 하이퍼파라미터 최적화를 위한 목적 함수로, optuna 라이브러리에서 사용
    
    storage = RDBStorage(
        url=hook.get_uri().replace("/postgres", "/optuna"), # RDBStorage: optuna의 스토리지
        # url: optuna의 스토리지 URL
        # hook.get_uri(): PostgresHook의 URI
        # "/postgres": PostgresHook의 URI에서 "/postgres"를 "/optuna"로 대체
    )
    
    study = optuna.create_study( # optuna.create_study: optuna의 study 생성
        study_name="hospital_model",
        direction="maximize", # maximize: objective 함수의 값(f1 score)을 최대화하는 방향으로 최적화
        storage=storage, # storage: optuna의 스토리지
        load_if_exists=True # load_if_exists: study가 이미 존재할 경우 불러올지 여부
    )
    
    study.optimize(objective, n_trials=10) # study.optimize: objective 함수를 최적화
    # n_trials: 몇 번의 시도를 통해 최적화할지
    
    best_params = study.best_params # study.best_params: 최적의 하이퍼파라미터
    best_metric = study.best_value # study.best_value: 최적의 지표 값
    
    print("Best params: ", best_params)
    print("Best metrics: ", best_metric)
    
    model = RandomForestClassifier(**best_params)
    model.fit(x_train, y_train)
    
    validation_score = f1_score(y_valid, model.predict(x_valid),average="micro")
    print(
        "validation score: ", validation_score
        # average: f1_score의 계산 방법 지정. "micro"는 전체 데이터에 대한 정확도를 계산
    )
    
    metrics = {
        "f1_score": validation_score
    }
    
    with mlflow.start_run():
        mlflow.log_params(best_params) # mlflow.log_params: 모델의 하이퍼파라미터를 저장
        mlflow.log_metrics(metrics) # mlflow.log_metrics: 모델의 성능 지표를 저장
        model_info = mlflow.sklearn.log_model(model, "model") # mlflow.sklearn.log_model: 모델을 저장
        
    context["ti"].xcom_push( # context["ti"].xcom_push: Airflow의 XCom을 통해 데이터를 전달
                             # ti: task instance의 약자. 현재 task의 인스턴스를 나타냄
                             # xcom_push: 데이터를 다른 task로 전달
        key="run_id", # key: 데이터를 전달할 때 사용할 키
        value=model_info.run_id) # model_info.run_id: mlflow에서 모델을 저장할 때 생성되는 고유 ID
    # run_id: mlflow에서 모델을 저장할 때 생성되는 고유 ID
    # run_id를 XCom을 통해 다음 task로 전달
    # XCom: Airflow의 task 간 데이터를 전달하는 메커니즘
    # XCom은 task의 output을 다른 task의 input으로 전달할 때 사용
    
    context["ti"].xcom_push(
        key="model_uri", 
        value=model_info.model_uri # model_info.model_uri: 모델의 저장 경로
    )
    
    context["ti"].xcom_push(
        key="eval_metric",
        value="f1_score"
    )
    
    print(
        f"Done training model. run_id: {model_info.run_id}, model_uri: {model_info.model_uri}"
    )
    
    # train_fn_for_hospital 함수의 역할
    # 1. PostgresHook을 통해 데이터를 불러와 전처리
    # 2. RandomForestClassifier 모델을 학습하고 최적의 하이퍼파라미터를 찾음
    # 3. mlflow를 통해 모델의 메타데이터와 성능 지표를 저장
    # 4. XCom을 통해 모델의 run_id와 저장 경로를 다음 task로 전달


def create_model_version(model_name: str, **context):
    run_id = context['ti'].xcom_pull(key="run_id") # context["ti"].xcom_pull: Airflow의 XCom을 통해 데이터를 받아옴
    model_uri = context['ti'].xcom_pull(key="model_uri")
    eval_metric = context['ti'].xcom_pull(key="eval_metric")
    
    client = MlflowClient()
    # MlflowClient: mlflow를 사용해 모델을 관리하는 클래스
    # mlflow를 사용해 모델을 관리하기 위해 MlflowClient를 생성
    # MlflowClient를 통해 mlflow 서버에 접속하여 모델을 관리할 수 있음
    
    try:
        client.create_registered_model(model_name) 
        # create_registered_model: 모델 등록
        # model_name: 등록할 모델의 이름
        # 모델을 등록할 때는 모델의 이름만 지정하면 됨
        # 모델의 버전은 mlflow에서 자동으로 관리
        # 모델을 등록하면 mlflow에서 모델의 메타데이터를 관리할 수 있음
        # 모델의 메타데이터: 모델의 이름, 설명, 태그, 버전 등
        # 모델의 메타데이터를 통해 모델을 검색하거나 관리할 수 있음
    except Exception as e:
        print("Model already exists.")
        # 모델이 이미 등록되어 있을 경우 예외 처리
        
    current_metric = client.get_run(run_id).data.metrics[eval_metric] 
    # 현재 모델의 성능 지표를 가져옴
    # get_run: mlflow에서 실험을 가져오는 함수
    # run_id: 가져올 run의 ID
    # data.metrics: run의 메트릭을 가져오는 속성
    # eval_metric: 가져올 메트릭의 이름. 여기서는 f1_score
    model_source = RunsArtifactRepository.get_underlying_uri(model_uri)
    # RunsArtifactRepository: mlflow에서 모델의 저장 경로를 관리하는 클래스
    # RunsArtifactRepository.get_underlying_uri: 모델의 저장 경로를 가져오는 함수
    # model_uri: 모델의 저장 경로
    # 모델의 저장 경로를 가져와서 model_source에 저장
    # model_source: 모델의 저장 경로
    # RunsArtifactRepository.get_underlying_uri의 리턴 타입: str
    model_version = client.create_model_version( # create_model_version: 모델의 버전을 생성
        model_name, model_source, run_id, description=f"{eval_metric}: {current_metric}"
    ) 
    # model_name: 모델의 이름
    # model_source: 모델의 저장 경로
    # run_id: 모델의 run ID
    # description: 모델의 설명
    # 모델의 버전을 생성하면 mlflow에서 모델의 버전을 관리할 수 있음
    # client.create_model_version의 리턴 타입: ModelVersion
    
    context['ti'].xcom_push(key="model_version", value=model_version.version)
    print(f"Done creating model version. model_version: {model_version.version}")
    
    # create_model_version의 역할
    # 1. mlflow에서 모델을 등록하고 모델의 버전을 생성
    # 2. 모델의 성능 지표를 가져와 모델의 설명에 추가
    # 3. XCom을 통해 모델의 버전을 다음 task로 전달

def transition_model_stage(model_name: str, **context):
    version = context['ti'].xcom_pull(key="model_version")
    eval_metric = context['ti'].xcom_pull(key="eval_metric")
    
    client = MlflowClient()
    production_model = None # production_model: 현재 운영 중인 모델
    current_model = client.get_model_version(model_name, version)
    # get_model_version: 모델의 버전을 가져오는 함수
    # model_name: 가져올 모델의 이름
    # version: 가져올 모델의 버전
    
    filter_string = f"name='{current_model.name}'"
    # filter_string: 모델을 검색할 때 사용할 필터
    
    results = client.search_model_versions(filter_string)
    # search_model_versions: 모델을 검색하는 함수
    
    for mv in results:
        # results는 ModelVersion의 검색 결과 리스트
        if mv.current_stage == "Pruduction": # 현재 운영 중인 모델을 찾음
            production_model = mv # production_model에 현재 운영 중인 모델을 저장
            
        if production_model is None: # 현재 운영 중인 모델이 없을 경우
            client.transition_model_version_stage( # transition_model_version_stage: 모델의 스테이지를 변경하는 함수
                current_model.name, # current_model.name: 현재 모델의 이름
                current_model.version, # current_model.version: 현재 모델의 버전
                "Production" # "Production": 모델의 스테이지를 운영 중으로 변경
            ) # 현재 모델을 운영 중으로 변경
            production_model = current_model # production_model에 현재 모델을 저장
            
        else: # 현재 운영 중인 모델이 있을 경우
            current_metric = client.get_run( # client.get_run: mlflow에서 실험을 가져오는 함수
                current_model.run_id # current_model.run_id: 현재 모델의 run ID
            ).data.metrics[eval_metric] # 현재 모델의 성능 지표를 가져옴
            production_metric = client.get_run(production_model.run_id).data.metrics[eval_metric] # 운영 중인 모델의 성능 지표를 가져옴
            
            if current_metric > production_metric: # 현재 모델의 성능이 운영 중인 모델의 성능보다 좋을 경우
                client.transition_model_version_stage( # 모델의 스테이지를 변경
                    current_model.name, 
                    current_model.version,
                    "Production",
                    archive_existing_versions=True, # 기존 버전을 아카이브할지 여부
                )
                production_model = current_model
                
        context['ti'].xcom_push(key="production_version", value=production_model.version)
        print(
            f"Done deploying model. production_version: {production_model.version}"
        )