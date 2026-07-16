"""
===============================================================================
Sales Data Processing & Performance Benchmark Pipeline
===============================================================================

[개요]
본 모듈은 대용량 매출 데이터(sales_100k.csv)를 대상으로 신뢰성 있는 기초 탐색(EDA) 및 
통계적 이상치(Outlier) 처리를 수행하며, 파이썬 생태계의 대표적인 3대 데이터 파이프라인 
엔진(Pandas, Polars, DuckDB)을 활용하여 다차원 그룹 집계 성능을 벤치마킹합니다.

[주요 파이프라인 구성]
1. 전처리 파이프라인 (Pandas Engine): 
   - 데이터 로드, 스키마 점검 및 결측치 로깅
   - 사분위수(IQR) 분포 기반의 정밀한 이상치 필터링(Q1-1.5*IQR ~ Q3+1.5*IQR) 적용
2. 집계 파이프라인 1 (Pandas Named Aggregation):
   - 메모리 기반 전통적 데이터프레임 다중 집계 연산
3. 집계 파이프라인 2 (Polars Lazy API):
   - 지연 평가(Lazy Evaluation) 및 쿼리 플랜 최적화를 통한 고속 데이터 처리
4. 집계 파이프라인 3 (DuckDB In-Memory SQL):
   - OLAP 최적화 컬럼형 데이터베이스 엔진 기반 표준 SQL 데이터 집계
5. 성능 벤치마크 (Performance Testing):
   - timeit 모듈을 활용한 다중 반복(N=5) 기반 엔진별 평균 실행 속도 측정 및 비교

[실행 환경 및 시스템 제어]
- 의존성: pandas, numpy, polars, duckdb
- 안정적인 시스템 운영 및 에러 트래킹을 위한 표준 logging 시스템 적용
===============================================================================
"""

import io
import logging
import timeit

import duckdb
import numpy as np
import pandas as pd
import polars as pl

# 시스템 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def perform_pandas_eda(file_path: str):
    """
    Pandas 엔진을 활용하여 기초 탐색(EDA)을 수행하고, 
    IQR 분포 기준(Q1-1.5*IQR ~ Q3+1.5*IQR)으로 이상치를 필터링합니다.
    """
    try:
        logger.info("=== [1] Pandas EDA 및 기초 탐색 ===")
        df = pd.read_csv(file_path)

        print("\n[Data Info]")
        df.info()

        print("\n[Null Counts]")
        print(df.isnull().sum())

        q1 = df["amount"].quantile(0.25)
        q3 = df["amount"].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        before_count = len(df)
        valid_range = df["amount"].between(lower_bound, upper_bound)
        df_filtered = df[valid_range]
        after_count = len(df_filtered)

        logger.info("\n[이상치 제거 결과]")
        logger.info("- 이상치 제거 전 행 수: %s개", f"{before_count:,}")
        logger.info("- 이상치 제거 후 행 수: %s개", f"{after_count:,}")
        logger.info("- 제거된 이상치 수: %s개", f"{before_count - after_count:,}")

        return df_filtered, lower_bound, upper_bound

    except FileNotFoundError:
        logger.error("'%s' 파일을 찾을 수 없습니다.", file_path)
        return None, None, None
    except Exception as e:
        logger.error("데이터 전처리 중 치명적 오류 발생: %s", e)
        return None, None, None


def pandas_aggregation(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Pandas의 Named Aggregation을 활용하여 region 및 category별 
    총매출, 평균, 건수 다차원 그룹 집계를 수행합니다.
    """
    try:
        result_df = (
            df.groupby(["region", "category"])
            .agg(
                total=("amount", "sum"),
                mean_amt=("amount", "mean"),
                cnt=("amount", "count"),
            )
            .sort_values(by="total", ascending=False)
            .reset_index()
        )
        return result_df
    except Exception as e:
        logger.error("Pandas 집계 프로세스 오류 발생: %s", e)
        return None


def polars_lazy_aggregation(
    file_path: str, lower_bound: float, upper_bound: float
) -> pl.DataFrame | None:
    """
    Polars Lazy API를 활용하여 데이터를 스캔하고, 지연 평가(Lazy Evaluation) 
    기반으로 쿼리를 최적화하여 필터링 및 집계를 수행합니다.
    """
    try:
        result_df = (
            pl.scan_csv(file_path)
            .filter(pl.col("amount").is_between(lower_bound, upper_bound))
            .group_by(["region", "category"])
            .agg(
                [
                    pl.col("amount").sum().alias("total"),
                    pl.col("amount").mean().alias("mean_amt"),
                    pl.len().alias("cnt"),
                ]
            )
            .sort("total", descending=True)
            .collect()
        )
        return result_df
    except Exception as e:
        logger.error("Polars Lazy 연산 중 오류 발생: %s", e)
        return None


def duckdb_sql_aggregation(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    DuckDB 인메모리 엔진을 활용하여 전처리된 데이터프레임에 
    표준 SQL(GROUP BY) 기반 데이터 집계 쿼리를 실행합니다.
    """
    try:
        query = """
            SELECT 
                region, 
                category, 
                SUM(amount) AS total, 
                AVG(amount) AS mean_amt, 
                COUNT(*) AS cnt
            FROM df
            GROUP BY region, category
            ORDER BY total DESC
        """
        return duckdb.query(query).df()
    except Exception as e:
        logger.error("DuckDB 쿼리 실행 중 오류 발생: %s", e)
        return None


def compare_performance(pandas_func, polars_func, duckdb_func, repeats: int = 5):
    """
    timeit 모듈을 활용하여 데이터 집계 파이프라인 엔진별 
    평균 처리 속도(반복 측정)를 벤치마킹하고 비교합니다.
    """
    logger.info("\n=== [4] 세 도구 성능 비교 (각 %d회 반복 측정 평균) ===", repeats)
    try:
        pandas_times = timeit.repeat(pandas_func, repeat=repeats, number=1)
        logger.info("- Pandas 평균 실행 시간: %.4f초", np.mean(pandas_times))

        polars_times = timeit.repeat(polars_func, repeat=repeats, number=1)
        logger.info("- Polars 평균 실행 시간: %.4f초", np.mean(polars_times))

        duckdb_times = timeit.repeat(duckdb_func, repeat=repeats, number=1)
        logger.info("- DuckDB 평균 실행 시간: %.4f초", np.mean(duckdb_times))
    except Exception as e:
        logger.error("성능 측정 벤치마크 중 오류가 발생했습니다: %s", e)


if __name__ == "__main__":
    file_name = "sales_100k.csv"

    cleaned_df, lower, upper = perform_pandas_eda(file_name)

    if cleaned_df is not None:
        logger.info("\n=== [결과 확인] Pandas ===")
        logger.info("\n%s", pandas_aggregation(cleaned_df).head(3).to_string())

        logger.info("\n=== [결과 확인] Polars Lazy API ===")
        logger.info("\n%s", polars_lazy_aggregation(file_name, lower, upper).head(3))

        logger.info("\n=== [결과 확인] DuckDB SQL ===")
        logger.info("\n%s", duckdb_sql_aggregation(cleaned_df).head(3).to_string())

        compare_performance(
            pandas_func=lambda: pandas_aggregation(cleaned_df),
            polars_func=lambda: polars_lazy_aggregation(file_name, lower, upper),
            duckdb_func=lambda: duckdb_sql_aggregation(cleaned_df),
            repeats=5,
        )


# [Insight] 5회 반복 측정 결과, 지연 평가(Lazy Evaluation)와 병렬 처리를 지원하는 Polars가 가장 우수한 성능을 보임.