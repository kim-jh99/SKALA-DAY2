"""
================================================================================
[프로그램 머리말]
- 프로그램명: [실습 2] 데이터 검증 파이프라인
- 순서: 1) 예외 처리 로드 → 2) Pydantic 정의 → 3) 검증 파이프라인 → 4) 저장/재검증
- 이해관계자: 데이터 파이프라인 엔지니어, 백엔드 서버 개발팀
================================================================================
"""

import csv
import json
import logging
from pydantic import BaseModel, Field, ValidationError

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ==============================================================================
# 1) 예외 처리 + 파일 읽기
# ==============================================================================
def safe_load_json(file_path: str):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"'{file_path}' 로딩 성공 (총 {len(data)}건)")
            return data
    except FileNotFoundError:
        logger.error(f"'{file_path}' 파일이 존재하지 않습니다.")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"'{file_path}' JSON 파싱 오류: {e}")
        return None
    finally:
        logger.info("로딩 종료")


# ==============================================================================
# 2) Pydantic v2 스키마 정의
# ==============================================================================
class SalesRecord(BaseModel):
    month: str = Field(min_length=1)
    region: str = Field(min_length=1)
    amount: float = Field(gt=0)
    category: str | None = Field(default=None)


# ==============================================================================
# 3) 검증 파이프라인 + 4) 결과 저장 + 재로딩 확인
# ==============================================================================
def main():
    # 3) 검증 파이프라인 실행
    target_file = 'Python_Practice2_Data.json'
    raw_data = safe_load_json(target_file)
    if not raw_data: return

    valid = []
    errors = []
    
    for i, row in enumerate(raw_data):
        if row.get('category') == "": row['category'] = None
            
        try:
            record = SalesRecord.model_validate(row)
            valid.append(record)
        except ValidationError as e:
            logger.warning(f"Row {i+1} 검증 실패: {e.errors()[0]['msg']}")
            errors.append({'row': i + 1, 'error': e.errors()})
            
    logger.info(f"검증 완료 - 정상: {len(valid)}건, 오류: {len(errors)}건")

    # 4) 결과 파일 저장 + 재로딩 확인
    valid_csv = 'valid_sales.csv'
    error_json = 'errors.json'
    
    if valid:
        with open(valid_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=valid[0].model_dump().keys())
            writer.writeheader()
            for record in valid:
                writer.writerow(record.model_dump())
        logger.info(f"정상 데이터 {len(valid)}건 저장 완료")

    if errors:
        with open(error_json, 'w', encoding='utf-8') as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        logger.info(f"오류 데이터 {len(errors)}건 저장 완료")

    # 재로딩 검증
    try:
        with open(valid_csv, 'r', encoding='utf-8') as f:
            reloaded = list(csv.DictReader(f))
            assert len(reloaded) == len(valid), "재로딩 건수 불일치!"
            logger.info("최종 검증 완료.")
    except Exception as e:
        logger.error(f"최종 검증 실패: {e}")

if __name__ == "__main__":
    main()