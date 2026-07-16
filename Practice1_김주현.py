"""
================================================================================
[프로그램 머리말]
- 프로그램명: 파이썬 자료구조 및 컬렉션 모듈 활용 매출 데이터 집계 프로그램
- 작성목적: Python_Practice1_Data.json 데이터를 읽어와 요구사항에 맞게 전처리 및 집계
- 이해관계자: 데이터 분석팀, 파이썬 백엔드 개발팀 (Python Code를 통한 Comm. 목적)
- 변경내역:
  * 2026.07.15 (v1.0): 최초 작성 및 컴프리헨션, Counter, defaultdict, Generator 적용
  * 2026.07.15 (v1.1): 실무 커뮤니케이션용 머리말/주석 추가 및 전역 예외(Error) 처리 반영
================================================================================
"""

import json
import sys
from collections import Counter, defaultdict

def load_sales_data(file_path):
    """
    [함수 설명]
    외부 JSON 파일을 안전하게 읽어와 파이썬 객체(List[dict])로 반환합니다.
    (예외/오류 처리 연습을 위해 발생 가능한 모든 파일, 파싱 에러를 대응합니다.)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            raw_text = file.read()
            
            # 'sales = ' 등의 불필요한 텍스트를 우회하여 순수 JSON 배열만 추출 (간결성)
            start_idx = raw_text.find('[')
            if start_idx == -1:
                raise ValueError("데이터 내에 유효한 JSON 시작점( '[' )이 없습니다.")
            
            clean_json_text = raw_text[start_idx:]
            return json.loads(clean_json_text)
            
    except FileNotFoundError:
        print(f"❌ [File Error] '{file_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ [Parse Error] JSON 데이터를 파싱하는 중 오류가 발생했습니다: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ [Unexpected Error] 데이터 로드 중 알 수 없는 오류가 발생했습니다: {e}")
        sys.exit(1)

def main():
    """
    [기능 설명]
    메인 집계 로직 실행부입니다.
    불필요한 반복문(for)을 지양하고 컴프리헨션과 내장 모듈로 코드를 간결하게 유지합니다.
    """
    sales_data = load_sales_data('Python_Practice1_Data.json')
    print(f"✅ 데이터 로드 성공 (총 {len(sales_data)}건의 거래 데이터)\n")

    # 전체 데이터 집계 과정에서 발생할 수 있는 데이터 결측, 타입 에러를 포괄적으로 예외 처리
    try:
        # =========================================================
        # [실습 1] 리스트/딕셔너리 컴프리헨션
        # =========================================================
        print("--- [실습 1] 컴프리헨션 필터링 및 집계 ---")
        
        # ① 리스트 컴프리헨션: 일반 for문 반복 지양, 코드 간결성 확보
        filtered_sales = [sale for sale in sales_data if sale.get("amount", 0) >= 1000]
        
        # ② 딕셔너리 컴프리헨션: 지역별 총매출 집계
        region_total = {
            region: sum(sale["amount"] for sale in filtered_sales if sale["region"] == region)
            for region in set(sale["region"] for sale in filtered_sales)
        }
        
        # [Assert 통과 확인] '서울' 데이터가 존재할 경우 총매출이 17670이 맞는지 강제 검증
        if "서울" in region_total:
            assert region_total["서울"] == 17670, "서울 지역의 총매출 계산 결과가 일치하지 않습니다!"
            
        print(f"1) 1000 이상 거래 수: {len(filtered_sales)}건")
        print(f"2) 지역별 총매출: {region_total}")


        # =========================================================
        # [실습 2] Counter + defaultdict
        # =========================================================
        print("\n--- [실습 2] Counter & defaultdict 활용 ---")
        
        # ① Counter: 직접 루프 카운팅 지양 (most_common 순서 정확도 보장)
        region_counter = Counter(sale["region"] for sale in sales_data)
        top_regions = region_counter.most_common()
        
        # ② defaultdict: 'if key not in dict' 반복 패턴 지양 및 간결성 확보
        category_amounts = defaultdict(list)
        for sale in sales_data:
            category_amounts[sale["category"]].append(sale["amount"])
            
        print(f"1) 지역별 거래 건수(순위별): {top_regions}")
        print(f"2) 추출된 카테고리 목록: {list(category_amounts.keys())}")


        # =========================================================
        # [실습 3] 제너레이터 메모리 비교
        # =========================================================
        print("\n--- [실습 3] 제너레이터 메모리 비교 ---")
        
        list_version = [sale for sale in sales_data if sale.get("amount", 0) > 1000]
        # list() 변환 없이 제너레이터(yield) 원형 유지
        generator_version = (sale for sale in sales_data if sale.get("amount", 0) > 1000)
        
        size_list = sys.getsizeof(list_version)
        size_gen = sys.getsizeof(generator_version)
        
        # [Assert 통과 확인] 제너레이터 메모리가 리스트보다 무조건 작음을 검증
        assert size_gen < size_list, "제너레이터의 메모리 이점이 확인되지 않았습니다!"
        
        print(f"- List 메모리: {size_list} Bytes")
        print(f"- Generator 메모리: {size_gen} Bytes (✅ 검증 완료: 메모리 절약 확인)")


        # =========================================================
        # [실습 4] 종합 - 월별 카테고리 매출 집계 & Top 3 정렬
        # =========================================================
        print("\n--- [실습 4] 월별 카테고리 집계 및 Top 3 정렬 ---")
        
        # ① (month, category) 복합 키 그룹핑 (defaultdict 활용)
        monthly_cat_sales = defaultdict(int)
        for sale in sales_data:
            # 존재하지 않는 키 참조 시 KeyError 방지를 위해 .get() 활용 (예외처리 연습)
            key = (sale.get("month", "Unknown"), sale.get("category", "Unknown"))
            monthly_cat_sales[key] += sale.get("amount", 0)
            
        # 가독성을 위한 포맷팅 (딕셔너리 컴프리헨션)
        formatted_sales = {
            f"{month} / {category}": total
            for (month, category), total in monthly_cat_sales.items()
        }
        
        # ② [정렬 정확] 각 카테고리별 상위 3개 거래 금액(내림차순 정렬)
        top3_category_sales = {
            category: sorted(amounts, reverse=True)[:3]
            for category, amounts in category_amounts.items()
        }
        
        print("1) 월별/카테고리별 총매출 (일부):")
        for k, v in list(formatted_sales.items())[:3]:
            print(f"   - {k}: {v}원")
            
        print("\n2) 카테고리별 상위 3개 거래 금액(내림차순):")
        for cat, top3 in top3_category_sales.items():
            print(f"   - {cat}: {top3}")
            
        print("\n🎉 모든 실습, Assert 검증, 예외 처리(Error Handling)를 성공적으로 통과했습니다!")

    # 실무 예외 처리(Error Handling) 연습: 데이터 조작 중 발생할 수 있는 에러 낚아채기
    except KeyError as e:
        print(f"\n❌ [Data Error] 딕셔너리에 필요한 키(Key)가 누락되었습니다: {e}")
    except TypeError as e:
        print(f"\n❌ [Type Error] 숫자 연산 불가 등 데이터 타입이 맞지 않습니다: {e}")
    except AssertionError as e:
        print(f"\n❌ [Validation Error] 실습 채점 기준 검증에 실패했습니다: {e}")

# 파이썬 스크립트 실행의 표준(Comm.) 진입점
if __name__ == "__main__":
    main()