"""
===============================================================================
[실습 4] 데이터 시각화, 통계 검정 및 머신러닝 파이프라인 구축
===============================================================================

작성자 : 울산 3반 김주현
작성일 : 2026-07-16

[개요]
    Practice3 의 전처리 함수 perform_pandas_eda를
    직접 호출하여 IQR 이상치가 제거된 데이터를 확보한 뒤, 이를 입력으로
    아래 4단계 분석 파이프라인을 순차 실행합니다.

    1. EDA 시각화    : 2x2 서브플롯(히스토그램/박스플롯/월별추이/상관히트맵)
    2. 통계 검정     : t-test(서울-부산 매출 차이) + 카이제곱(지역-카테고리 독립성)
    3. ML 파이프라인 : ColumnTransformer+Pipeline 구성 후 fit·predict·score, joblib 저장
    4. 인터랙티브    : Plotly 막대 차트 생성 후 HTML 저장

[의존성]
    pandas, numpy, matplotlib, seaborn, plotly, scipy, scikit-learn, joblib
    (practice3 import 시 duckdb, polars 도 함께 필요)
===============================================================================
"""

import logging

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Practice 3의 전처리 함수 재사용 (IQR 이상치 제거 로직을 중복 작성하지 않음)
from Practice3_김주현 import perform_pandas_eda

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_and_preprocess_data(file_path):
    """실습 3의 perform_pandas_eda를 호출해 IQR 이상치가 제거된 DataFrame을 반환합니다."""
    try:
        df_filtered, lower, upper = perform_pandas_eda(file_path)
        if df_filtered is None:
            logger.error("전처리 실패: 실습 3 함수가 데이터를 반환하지 못했습니다.")
            return None
        logger.info(
            "전처리 완료: %d개 행 확보 (제거 범위: %.2f ~ %.2f)",
            len(df_filtered), lower, upper,
        )
        # 이후 단계에서 컬럼을 변형하므로 원본 슬라이스 대신 복사본 사용
        return df_filtered.copy()
    except Exception as e:
        logger.error("데이터 전처리 중 오류 발생: %s", e)
        return None


def perform_eda(df):
    """2x2 서브플롯으로 EDA 시각화 4종을 하나의 figure에 작성합니다."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # (1) 매출 분포 — 히스토그램 + KDE
        sns.histplot(df["amount"], kde=True, ax=axes[0, 0])
        axes[0, 0].set_title("Amount Distribution")

        # (2) 카테고리별 매출 — 박스플롯
        sns.boxplot(x="category", y="amount", data=df, ax=axes[0, 1])
        axes[0, 1].set_title("Amount by Category")

        # (3) 월별 매출 추이 — 라인
        df["order_date"] = pd.to_datetime(df["order_date"])
        monthly = df.set_index("order_date").resample("ME")["amount"].sum()
        monthly.plot(ax=axes[1, 0])
        axes[1, 0].set_title("Monthly Revenue Trend")

        # (4) 수치형 변수 상관관계 — 히트맵
        numeric_df = df.select_dtypes(include=[np.number])
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=axes[1, 1])
        axes[1, 1].set_title("Correlation Heatmap")

        plt.tight_layout()
        plt.show()
        logger.info("EDA 시각화 완료 (2x2 서브플롯 1개 figure).")
    except Exception as e:
        logger.error("시각화 중 오류 발생: %s", e)


def perform_stats(df):
    """t-test(서울 vs 부산)와 카이제곱(지역 vs 카테고리)을 수행하고 p-value를 해석합니다."""
    try:
        # --- t-test : 서울 vs 부산 평균 매출 차이 ---
        seoul = df.loc[df["region"] == "서울", "amount"]
        busan = df.loc[df["region"] == "부산", "amount"]
        t_stat, p_val = stats.ttest_ind(seoul, busan)
        logger.info("[t-test] 서울 vs 부산 | t=%.4f, p-value=%.4f", t_stat, p_val)
        if p_val < 0.05:
            logger.info("  해석: 두 지역 평균 매출 차이가 통계적으로 유의미함 (p < 0.05)")
        else:
            logger.info("  해석: 두 지역 평균 매출 차이가 유의미하지 않음 (p >= 0.05)")

        # --- 카이제곱 : 지역 x 카테고리 독립성 ---
        contingency = pd.crosstab(df["region"], df["category"])
        chi2, p_chi, dof, _ = stats.chi2_contingency(contingency)
        logger.info(
            "[카이제곱] 지역 x 카테고리 | chi2=%.4f, dof=%d, p-value=%.4f",
            chi2, dof, p_chi,
        )
        if p_chi < 0.05:
            logger.info("  해석: 지역과 카테고리는 서로 관련 있음 (독립 아님, p < 0.05)")
        else:
            logger.info("  해석: 지역과 카테고리는 서로 독립적임 (p >= 0.05)")
    except Exception as e:
        logger.error("통계 검정 중 오류 발생: %s", e)


def run_ml_pipeline(df):
    """ColumnTransformer+Pipeline을 구성해 fit·predict·score 실행 후 joblib으로 저장합니다."""
    try:
        num_features = ["quantity", "unit_price", "customer_age"]
        cat_features = ["region", "category", "payment_method", "customer_gender"]

        X = df[num_features + cat_features]
        y = df["amount"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 전처리(스케일링 + 원-핫)와 모델을 하나의 Pipeline 객체로 묶음
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ]
        )
        pipeline = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("regressor", RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1)),
            ]
        )

        # fit -> predict -> score
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        r2 = pipeline.score(X_test, y_test)
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        logger.info("모델 학습/평가 완료 | R2=%.4f, RMSE=%.2f", r2, rmse)

        # 학습된 파이프라인 전체를 파일로 저장
        joblib.dump(pipeline, "model_pipeline.pkl")
        logger.info("모델 저장 완료 (model_pipeline.pkl)")
        # 저장한 모델 재로딩 후 동작 검증
        loaded_pipeline = joblib.load("model_pipeline.pkl")
        reload_r2 = loaded_pipeline.score(X_test, y_test)
        logger.info("모델 재로딩 완료 | 재로딩 후 R2=%.4f", reload_r2)
    except Exception as e:
        logger.error("파이프라인 구성/학습 중 오류 발생: %s", e)


def save_plotly_chart(df):
    """지역·카테고리별 총매출을 Plotly 막대 차트로 만들어 HTML로 저장합니다."""
    try:
        grouped = df.groupby(["region", "category"])["amount"].sum().reset_index()
        fig = px.bar(
            grouped,
            x="category",
            y="amount",
            color="region",
            barmode="group",
            title="지역·카테고리별 총매출",
        )
        fig.write_html("interactive_chart.html")
        logger.info("Plotly 인터랙티브 차트 저장 완료 (interactive_chart.html)")
    except Exception as e:
        logger.error("Plotly 차트 저장 중 오류 발생: %s", e)


if __name__ == "__main__":
    df_cleaned = load_and_preprocess_data("sales_100k.csv")

    if df_cleaned is not None:
        perform_eda(df_cleaned)
        perform_stats(df_cleaned)
        run_ml_pipeline(df_cleaned)
        save_plotly_chart(df_cleaned)
        logger.info("실습 4 모든 과정 완료")
    else:
        logger.error("데이터 로드 실패로 실습 4를 종료합니다.")
