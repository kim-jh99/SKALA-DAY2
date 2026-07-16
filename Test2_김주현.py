"""
================================================================================
[프로그램 머리말]
- 과제명 : SKALA 종합실습2 - 서울시 상권분석 데이터 분석 및 ML 파이프라인 구축
- 데이터 : 서울시 상권분석서비스(추정매출-상권) CSV (cp949 인코딩)
- 주요기능
    2. 사용할 칼럼 선택
    3. 서비스_업종_코드_명별 당월_매출_금액 합계 상위 10개 추출
    4. 연령대(10/20/30) 매출 합계 bar 그래프 저장
    5. 수치형/범주형 파이프라인 구성 후 하나로 결합
    6. RandomForest 회귀 모델 학습 및 성능 평가
- 제출자 : 김주현
- 작성일 : 2026-07-16

[변경내역]
    2026-07-16  최초 작성
================================================================================
"""

import logging

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 그래프 한글 깨짐 방지 (macOS 기준)
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

FILE_PATH = "서울시 상권분석서비스(추정매출-상권).csv"


# ==============================================================================
# [2] 사용할 칼럼 지정 및 데이터 로드
# ==============================================================================
def load_selected_data(file_path):
    """CSV를 cp949로 읽고, 과제에서 지정한 칼럼만 선택해 반환합니다."""
    selected_columns = [
        "상권_코드_명",
        "상권_구분_코드_명",
        "서비스_업종_코드",
        "서비스_업종_코드_명",
        "당월_매출_금액",
        "남성_매출_금액",
        "여성_매출_금액",
        "연령대_10_매출_금액",
        "연령대_20_매출_금액",
        "연령대_30_매출_금액",
    ]
    try:
        df = pd.read_csv(file_path, encoding="cp949", usecols=selected_columns)
        logger.info("데이터 로드 완료: %d행, %d컬럼", df.shape[0], df.shape[1])
        return df
    except FileNotFoundError:
        logger.error("'%s' 파일을 찾을 수 없습니다.", file_path)
        return None
    except Exception as e:
        logger.error("데이터 로드 중 오류 발생: %s", e)
        return None


# ==============================================================================
# [3] 그룹 / 정렬 - 업종별 당월 매출 합계 상위 10개
# ==============================================================================
def get_top10_by_industry(df):
    """서비스_업종_코드_명별 당월_매출_금액 합계를 내림차순 정렬해 상위 10개를 반환합니다."""
    try:
        top10 = (
            df.groupby("서비스_업종_코드_명")["당월_매출_금액"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        logger.info("업종별 매출 상위 10개 추출 완료")
        print("\n[업종별 당월 매출 합계 TOP 10]")
        print(top10.to_string())
        return top10
    except Exception as e:
        logger.error("그룹/정렬 처리 중 오류 발생: %s", e)
        return None


# ==============================================================================
# [4] 그래프 / 파일 저장 - 연령대별 매출 합계 bar 그래프
# ==============================================================================
def save_age_group_barchart(df, file_name="age_group_sales.png"):
    """연령대 10/20/30 매출 컬럼별 합계를 bar 그래프로 그려 파일로 저장합니다."""
    try:
        age_columns = ["연령대_10_매출_금액", "연령대_20_매출_금액", "연령대_30_매출_금액"]
        age_sums = df[age_columns].sum()

        plt.figure(figsize=(8, 6))
        age_sums.plot(kind="bar", color=["#4C72B0", "#55A868", "#C44E52"])
        plt.title("연령대별 매출 합계 (10대 / 20대 / 30대)")
        plt.ylabel("매출 금액 합계")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(file_name, dpi=100, bbox_inches="tight")
        plt.close()
        logger.info("연령대별 bar 그래프 저장 완료 (%s)", file_name)
    except Exception as e:
        logger.error("그래프 저장 중 오류 발생: %s", e)


# ==============================================================================
# [5] & [6] 파이프라인 구성 + 모델 학습 / 성능 확인
# ==============================================================================
def build_and_evaluate_model(df):
    """수치형/범주형 파이프라인을 결합해 모델을 학습하고 성능을 평가합니다."""
    try:
        numeric_features = ["연령대_10_매출_금액", "연령대_20_매출_금액", "연령대_30_매출_금액"]
        categorical_features = ["상권_구분_코드_명"]
        target = "당월_매출_금액"

        X = df[numeric_features + categorical_features]
        y = df[target]

        # [5-1] 수치형 파이프라인 : 결측치(중간값) + 스케일링
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        # [5-2] 범주형 파이프라인 : 결측치(missing) + 원핫인코딩
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        # [5-3] 두 파이프라인을 하나로 결합
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline, categorical_features),
            ]
        )

        # [5-4] 전처리 + 모델을 묶은 최종 파이프라인
        model_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
            ]
        )

        # [6] 학습 및 성능 평가
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model_pipeline.fit(X_train, y_train)
        y_pred = model_pipeline.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        logger.info("모델 학습 완료 | R2=%.4f, RMSE=%.2f", r2, rmse)

        return model_pipeline
    except Exception as e:
        logger.error("파이프라인 구성/학습 중 오류 발생: %s", e)
        return None


# ==============================================================================
# 메인 실행
# ==============================================================================
if __name__ == "__main__":
    df = load_selected_data(FILE_PATH)

    if df is not None:
        get_top10_by_industry(df)       # [3]
        save_age_group_barchart(df)     # [4]
        build_and_evaluate_model(df)    # [5] & [6]
        logger.info("종합실습2 모든 과정 완료")
    else:
        logger.error("데이터 로드 실패로 종합실습2를 종료합니다.")