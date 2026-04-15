
# local_test.ipynb 코드 설명

이 문서는 [local_test.ipynb](/c:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/notebooks/local_test.ipynb) 를 위에서부터 따라가며 설명한 내용입니다.  
노트북에서 호출하는 [preprocess_bpi2012.py](/c:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/src/preprocess_bpi2012.py) 의 함수도 함께 설명합니다.

## 노트북 전체 구조

- `1번째 셀`: 설명용 markdown
- `2번째 셀`: 프로젝트 루트 경로 잡기, import 경로 설정
- `3번째 셀`: 전처리 함수 import
- `4번째 셀`: XES 파일 읽기, 이벤트 테이블 생성, 요약 통계
- `5번째 셀`: 생성된 이벤트 테이블 미리보기
- `6번째 셀`: 전체 이벤트 테이블 CSV 저장
- `7번째 셀`: 로컬 개발용 case subset 추출, 요약 통계
- `8번째 셀`: subset 미리보기
- `9번째 셀`: subset CSV 저장

## 1. Markdown 셀

참고 위치: [local_test.ipynb:3](/c:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/notebooks/local_test.ipynb:3)

이 셀은 실행 코드가 아니라, 이 노트북의 목적을 적어둔 셀입니다.

- `BPI 2012 local preprocessing test`
- 이 노트북이 하는 일:
- XES 로그를 event table로 바꾸기
- 로컬 개발용 small subset 만들기

즉, 이 노트북은 모델 학습이 아니라 전처리 첫 단계 검증용입니다.

## 2. 프로젝트 루트와 import 경로 설정 셀

참고 위치: [local_test.ipynb:20](/c:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/notebooks/local_test.ipynb:20)

```python
from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == 'notebooks' else Path.cwd().resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

print(PROJECT_ROOT)
```

이 셀은 노트북이 어디서 실행되든 `src` 폴더를 import할 수 있게 준비하는 코드입니다.

### `PROJECT_ROOT` 계산

- `Path.cwd()`는 현재 작업 폴더를 뜻합니다.
- 현재 폴더 이름이 `notebooks`면:
- 노트북 폴더 안에서 실행 중이라고 보고
- 상위 폴더를 프로젝트 루트로 잡습니다.
- 그렇지 않으면:
- 이미 프로젝트 루트에서 실행 중이라고 보고
- 현재 폴더를 그대로 씁니다.

즉, 이 한 줄은 노트북 실행 위치 차이를 흡수하려는 코드입니다.

```python
PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == 'notebooks' else Path.cwd().resolve()
```

### `sys.path` 추가

```python
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
```

이 코드는 파이썬 import 검색 경로에 프로젝트 루트를 추가합니다.

왜 필요하냐면 다음 셀에서 `from src.preprocess_bpi2012 import ...`를 쓰기 때문입니다.  
파이썬은 `sys.path`에 등록된 경로 안에서 모듈을 찾는데, 프로젝트 루트가 없으면 `src`를 못 찾을 수 있습니다.

### `print(PROJECT_ROOT)`

이 줄은 디버깅용입니다.

- 어떤 폴더를 루트로 잡았는지 확인
- 경로 설정이 틀렸을 때 바로 알 수 있음

즉, 이 셀은 전처리 자체보다 노트북 실행 환경을 안정화하는 셀입니다.

## 3. 전처리 함수 import 셀

참고 위치: [local_test.ipynb:36](/c:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/notebooks/local_test.ipynb:36)

```python
from src.preprocess_bpi2012 import load_bpi2012_events, sample_cases, save_event_table, summarize_event_table
```

이 셀은 노트북이 직접 복잡한 전처리 로직을 들고 있지 않고, `src/preprocess_bpi2012.py`에 있는 함수들을 가져와 쓰겠다는 뜻입니다.

가져오는 함수 4개 역할은 이렇습니다.

- `load_bpi2012_events`
- XES 파일을 읽어서 event table DataFrame 생성
- `sample_cases`
- 전체 이벤트 중 일부 case만 골라 subset 생성
- `save_event_table`
- DataFrame을 CSV로 저장
- `summarize_event_table`
- 데이터 개요 통계 계산

즉, 노트북은 orchestration 역할이고, 실제 로직은 모듈로 분리한 구조입니다.

## 4. XES 경로 설정, 전체 이벤트 테이블 생성, 요약 통계 셀

참고 위치: [local_test.ipynb:45](/c:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/notebooks/local_test.ipynb:45)

```python
XES_PATH = PROJECT_ROOT / 'data' / 'BPI_Challenge_2012.xes' / 'BPI_Challenge_2012.xes'
INTERIM_DIR = PROJECT_ROOT / 'data' / 'interim'
LOCAL_DIR = PROJECT_ROOT / 'sample_data' / 'bpi2012_local'
```

이 블록은 경로 변수 선언입니다.

- `XES_PATH`
- 원본 로그 파일 위치
- `INTERIM_DIR`
- 전체 전처리 결과 저장 폴더
- `LOCAL_DIR`
- 로컬 개발용 subset 저장 폴더

중요한 점은 문자열을 직접 붙이지 않고 `Path` 연산자 `/`를 썼다는 점입니다.  
이 방식은 운영체제 경로 구분자를 직접 처리할 필요가 없어서 가독성과 안정성이 좋습니다.

다음 블록:

```python
events = load_bpi2012_events(
    xes_path=XES_PATH,
    include_lifecycle=False,
    lifecycle_filter='COMPLETE',
)
```

여기가 이 노트북의 핵심 1단계입니다.

- `load_bpi2012_events(...)`를 호출해서
- `.xes` 파일을 읽고
- pandas DataFrame인 `events`를 만듭니다.

인자 의미는 다음과 같습니다.

- `xes_path=XES_PATH`
- 어떤 파일을 읽을지 지정
- `include_lifecycle=False`
- activity 이름에 lifecycle을 붙이지 않겠다는 뜻
- 예: `A_SUBMITTED`
- 붙이면 `A_SUBMITTED+COMPLETE` 같은 형식이 됨
- `lifecycle_filter='COMPLETE'`
- `COMPLETE` lifecycle 이벤트만 사용
- start/complete가 둘 다 있는 경우 complete만 남기는 전략

현재 목적에 맞는 꽤 합리적인 기본값입니다. 프로세스 이벤트 로그는 lifecycle이 섞여 있으면 item 종류가 불필요하게 많아질 수 있어서, baseline 단계에서는 `COMPLETE`만 쓰는 편이 깔끔합니다.

그 다음:

```python
summary = summarize_event_table(events)
summary
```

이 코드는 DataFrame 전체를 눈으로 훑기 전에, 전체 개요를 빠르게 파악하기 위한 요약 통계를 계산합니다.

## 5. `preprocess_bpi2012.py` 상단 import와 상수

참고 위치: [preprocess_bpi2012.py:1](/c:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/src/preprocess_bpi2012.py:1)

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import xml.etree.ElementTree as ET
import pandas as pd
```

이 블록은 모듈이 필요한 라이브러리를 불러오는 부분입니다.

- `Path`
- 파일 경로 처리
- `ET`
- XML 기반인 XES 파일 파싱
- `pandas`
- 테이블 데이터 처리
- `dataclass`
- 요약 결과를 구조화된 객체로 반환
- `Literal`
- `sample_cases`에서 허용 문자열 값을 타입으로 제한

다음 줄:

```python
XES_NS = {"xes": "http://www.xes-standard.org/"}
```

이건 XML namespace 설정입니다. XES는 namespace를 포함하므로 `trace`, `event` 같은 태그를 찾을 때 이 정보가 필요합니다.
## 6. `EventTableSummary` dataclass

참고 위치: [preprocess_bpi2012.py:14](/c:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/src/preprocess_bpi2012.py:14)

```python
@dataclass
class EventTableSummary:
    num_cases: int
    num_events: int
    num_activities: int
    min_case_length: int
    median_case_length: float
    max_case_length: int
```

이 클래스는 요약 통계를 담는 작은 컨테이너입니다.

왜 `dict` 대신 dataclass를 썼냐면:

- 필드 이름이 명확하고
- 출력이 깔끔하고
- IDE에서 보기 좋고
- 나중에 타입 추적도 쉽기 때문입니다

## 7. `_get_xes_value` 헬퍼 함수

참고 위치: [preprocess_bpi2012.py:24](/c:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/src/preprocess_bpi2012.py:24)

```python
def _get_xes_value(parent: ET.Element, key: str) -> str | None:
    for child in parent:
        if child.attrib.get("key") == key:
            return child.attrib.get("value")
    return None
```

이 함수는 XES XML 구조에서 특정 key의 value를 찾는 헬퍼입니다.

XES의 이벤트는 대략 이런 구조입니다.

```xml
<event>
  <string key="concept:name" value="A_SUBMITTED"/>
  <date key="time:timestamp" value="2011-10-01T..."/>
</event>
```

그래서 이벤트 안에서 `concept:name`, `lifecycle:transition`, `time:timestamp` 같은 key를 반복해서 찾게 되는데, 그 작업을 함수로 뺀 것입니다.

의미는 다음과 같습니다.

- 주어진 XML 노드의 자식들을 순회
- `key` 속성이 원하는 값과 같으면
- 그 자식의 `value`를 반환
- 못 찾으면 `None`

## 8. `load_bpi2012_events` 함수 전체 흐름

참고 위치: [preprocess_bpi2012.py:31](/c:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/src/preprocess_bpi2012.py:31)

이 함수가 가장 중요합니다. 역할은 XES 원본 로그를 pandas event table로 변환하는 것입니다.

함수 시그니처:

```python
def load_bpi2012_events(
    xes_path: str | Path,
    activity_key: str = "concept:name",
    lifecycle_key: str = "lifecycle:transition",
    timestamp_key: str = "time:timestamp",
    include_lifecycle: bool = False,
    lifecycle_filter: str | None = None,
) -> pd.DataFrame:
```

설계 포인트는 BPI 2012에 맞춘 기본값을 주되, 나중에 실험 변경이 가능하게 열어뒀다는 점입니다.

- `activity_key`: activity를 어떤 key에서 읽을지
- `lifecycle_key`: lifecycle을 어디서 읽을지
- `timestamp_key`: timestamp key
- `include_lifecycle`: activity 이름에 lifecycle 붙일지 여부
- `lifecycle_filter`: 특정 lifecycle만 필터링할지 여부

초반 입력 검증:

```python
xes_path = Path(xes_path)
if not xes_path.exists():
    raise FileNotFoundError(f"XES file not found: {xes_path}")
```

- 문자열이 와도 `Path`로 통일
- 파일이 없으면 즉시 실패
- 조용히 잘못된 결과를 만드는 대신 명시적으로 에러를 냅니다

다음 준비 단계:

```python
rows: list[dict[str, object]] = []
tree = ET.parse(xes_path)
root = tree.getroot()
```

여기서 XML 파일을 읽고 루트 노드를 잡습니다. `rows`는 나중에 DataFrame으로 바꿀 row 리스트입니다.

### trace 단위 순회

```python
for trace in root.findall("xes:trace", XES_NS):
    case_id = _get_xes_value(trace, "concept:name")
    if not case_id:
        continue
```

- XES에서 `trace`는 하나의 case
- trace의 `concept:name`은 case id

### event 단위 순회

```python
for event in trace.findall("xes:event", XES_NS):
    activity = _get_xes_value(event, activity_key)
    lifecycle = _get_xes_value(event, lifecycle_key)
    timestamp = _get_xes_value(event, timestamp_key)
```

각 case 내부에서 이벤트를 하나씩 읽어 필요한 필드만 뽑아냅니다.

### 필수값 필터

```python
if not activity or not timestamp:
    continue
```

activity나 timestamp가 없으면 sequential data로 쓰기 어려워서 건너뜁니다.

### lifecycle 필터

```python
if lifecycle_filter is not None:
    if lifecycle is None or lifecycle.lower() != lifecycle_filter.lower():
        continue
```

현재 노트북에서 `COMPLETE`만 남게 만드는 핵심 로직입니다.

### activity 이름 결정

```python
activity_name = activity
if include_lifecycle and lifecycle:
    activity_name = f"{activity}+{lifecycle}"
```

- 기본: `A_SUBMITTED`
- 옵션 활성화: `A_SUBMITTED+COMPLETE`

### row 추가와 후처리

```python
rows.append({
    "case_id": str(case_id),
    "activity": str(activity_name),
    "lifecycle": None if lifecycle is None else str(lifecycle),
    "timestamp": timestamp,
})
```

이후 DataFrame으로 바꾼 다음 아래 순서로 후처리합니다.

- `pd.to_datetime(..., utc=True, errors="coerce")`로 datetime 변환
- `dropna(...)`로 결측 제거
- `drop_duplicates(...)`로 완전 중복 제거
- `sort_values(["case_id", "timestamp", "activity"])`로 정렬
- `groupby("case_id").cumcount()`로 `event_idx` 생성

정리하면 `load_bpi2012_events`는 XML 로그를 읽고, 필요한 필드만 뽑고, 필터링/정렬해서 분석 가능한 event table DataFrame으로 만드는 함수입니다.
## 9. `summarize_event_table` 함수

참고 위치: [preprocess_bpi2012.py:92](/c:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/src/preprocess_bpi2012.py:92)

```python
def summarize_event_table(events: pd.DataFrame) -> EventTableSummary:
    if events.empty:
        return EventTableSummary(0, 0, 0, 0, 0.0, 0)
```

빈 데이터 방어 처리입니다.

```python
case_lengths = events.groupby("case_id").size()
```

case별 이벤트 개수를 계산합니다.

반환부:

```python
return EventTableSummary(
    num_cases=int(events["case_id"].nunique()),
    num_events=int(len(events)),
    num_activities=int(events["activity"].nunique()),
    min_case_length=int(case_lengths.min()),
    median_case_length=float(case_lengths.median()),
    max_case_length=int(case_lengths.max()),
)
```

전체 데이터의 규모와 시퀀스 길이 분포를 빠르게 확인하기 위한 함수입니다.

## 10. `events.head(10)` 셀

참고 위치: [local_test.ipynb:65](/c:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/notebooks/local_test.ipynb:65)

```python
events.head(10)
```

이 셀은 요약 통계 다음에 실제 row를 눈으로 확인하는 용도입니다.

보는 포인트는 보통 다음과 같습니다.

- `case_id`가 잘 들어왔는지
- `activity` 값이 기대한 이름인지
- `lifecycle`이 `COMPLETE`만 남았는지
- `timestamp`가 datetime으로 변환되었는지
- `event_idx`가 0부터 순서대로 붙는지

즉, `summary`가 전체 스케일 체크라면 `head(10)`은 row-level sanity check입니다.

## 11. 전체 이벤트 테이블 저장 셀

참고 위치: [local_test.ipynb:74](/c:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/notebooks/local_test.ipynb:74)

```python
full_events_path = save_event_table(events, INTERIM_DIR / 'bpi2012_events_complete_only.csv')
print(full_events_path)
```

이 셀은 메모리 안의 DataFrame을 파일로 저장합니다.

호출되는 함수는 [preprocess_bpi2012.py:145](/c:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/src/preprocess_bpi2012.py:145)의 `save_event_table`입니다.

```python
def save_event_table(events: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(output_path, index=False, encoding="utf-8")
    return output_path
```

이 함수는 3단계입니다.

- `output_path`를 `Path`로 통일
- 상위 폴더 없으면 자동 생성
- CSV 저장 후 저장 경로 반환

## 12. local subset 생성 셀

참고 위치: [local_test.ipynb:84](/c:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/notebooks/local_test.ipynb:84)

```python
N_LOCAL_CASES = 500
local_events = sample_cases(events, n_cases=N_LOCAL_CASES, strategy='earliest')
local_summary = summarize_event_table(local_events)
local_summary
```

이 셀은 로컬 개발용 subset을 만드는 핵심입니다.

- `N_LOCAL_CASES = 500`: 샘플링할 case 수
- `sample_cases(...)`: 전체 이벤트 테이블에서 case 단위로 subset 추출
- `summarize_event_table(local_events)`: subset 규모를 다시 확인

이 함수는 [preprocess_bpi2012.py:107](/c:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/src/preprocess_bpi2012.py:107)에 있습니다.

함수 시그니처:

```python
def sample_cases(
    events: pd.DataFrame,
    n_cases: int,
    strategy: Literal["earliest", "latest", "random"] = "earliest",
    random_state: int = 42,
) -> pd.DataFrame:
```

핵심 흐름은 다음과 같습니다.

```python
case_starts = (
    events.groupby("case_id", as_index=False)["timestamp"]
    .min()
    .rename(columns={"timestamp": "case_start"})
)
```

이 코드는 case별 시작 시점을 계산합니다.

그 다음 전략에 따라 case id를 선택합니다.

- `earliest`: 가장 먼저 시작한 case부터 N개
- `latest`: 가장 늦게 시작한 case부터 N개
- `random`: case 단위 랜덤 샘플

선택 후에는 다음처럼 subset을 만듭니다.

```python
sampled = events[events["case_id"].isin(selected_case_ids)].copy()
sampled = sampled.sort_values(["case_id", "timestamp", "activity"]).reset_index(drop=True)
sampled["event_idx"] = sampled.groupby("case_id").cumcount()
return sampled
```

즉, event를 랜덤으로 잘라내는 게 아니라 case 전체를 유지한 채 일부 case만 뽑는 방식입니다.

## 13. local subset 미리보기 셀

참고 위치: [local_test.ipynb:96](/c:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/notebooks/local_test.ipynb:96)

```python
local_events.head(10)
```

subset 버전의 row-level 확인입니다.

여기서 볼 포인트:

- subset에서도 정렬이 유지되는지
- `event_idx`가 다시 0부터 붙는지
- case가 온전하게 들어왔는지

## 14. local subset 저장 셀

참고 위치: [local_test.ipynb:105](/c:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/notebooks/local_test.ipynb:105)

```python
local_events_path = save_event_table(local_events, LOCAL_DIR / 'bpi2012_events_complete_only_500_cases.csv')
print(local_events_path)
```

이 셀은 로컬 개발용 subset을 실제 파일로 저장합니다.

저장 경로를 `sample_data/bpi2012_local/`로 잡은 이유는:

- 전체 데이터와 분리
- 로컬 테스트 결과라는 의미가 분명
- small sample 버전을 여러 개 만들기 쉬움

## 15. 이 노트북의 설계 철학

이 노트북은 한 번에 다 처리하는 거대한 셀이 아니라, 단계별 셀로 분리되어 있습니다.

분리 기준:

- 환경 설정
- 함수 import
- 전체 데이터 생성
- 눈으로 확인
- 파일 저장
- subset 생성
- subset 확인
- subset 저장

이렇게 나눈 이유는 디버깅 때문입니다.

- import 실패면 초반 셀에서 바로 멈춤
- XES 파싱 문제면 데이터 생성 셀에서 바로 드러남
- subset이 이상하면 subset 셀에서 바로 확인 가능

즉, 노트북다운 방식으로 중간 결과를 눈으로 확인하며 전처리를 진행하도록 만든 구조입니다.

## 16. 현재 이 코드가 만들어내는 데이터 형태

이 노트북 단계가 끝났을 때 `events`와 `local_events`는 대략 다음 컬럼을 가집니다.

- `case_id`: 원본 trace ID
- `activity`: 현재는 `concept:name`
- `lifecycle`: 원본 lifecycle 값
- `timestamp`: pandas datetime, UTC 기준
- `event_idx`: case 내부 순서 index

즉, 아직 SASRec용 `user item` 포맷은 아닙니다. 지금은 그 직전 단계인 정리된 event table입니다.

## 17. 다시 볼 때 핵심 흐름

1. 경로와 환경 준비  
참고: [local_test.ipynb:20](/c:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/notebooks/local_test.ipynb:20)

2. 전처리 함수 import  
참고: [local_test.ipynb:36](/c:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/notebooks/local_test.ipynb:36)

3. 원본 XES를 event table로 변환  
참고: [local_test.ipynb:45](/c:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/notebooks/local_test.ipynb:45)  
핵심 함수: [preprocess_bpi2012.py:31](/c:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/src/preprocess_bpi2012.py:31)

4. 요약 통계와 샘플 행으로 데이터 확인  
참고: [local_test.ipynb:55](/c:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/notebooks/local_test.ipynb:55), [local_test.ipynb:65](/c:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/notebooks/local_test.ipynb:65)

5. 전체 데이터 저장  
참고: [local_test.ipynb:74](/c:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/notebooks/local_test.ipynb:74)

6. case 단위 local subset 추출  
참고: [local_test.ipynb:84](/c:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/notebooks/local_test.ipynb:84)  
핵심 함수: [preprocess_bpi2012.py:107](/c:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/src/preprocess_bpi2012.py:107)

7. subset 확인 및 저장  
참고: [local_test.ipynb:96](/c:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/notebooks/local_test.ipynb:96), [local_test.ipynb:105](/c:/Users/hyewoo%20choi/Documents/99.%20%EB%8C%80%ED%95%99%EC%9B%90/0.%20%EB%85%BC%EB%AC%B8/git/time-aware-behavior-prediction/notebooks/local_test.ipynb:105)

이 흐름을 기준으로 보면, 현재 노트북은 원본 로그를 정리된 event table로 바꾸고, 로컬 개발용 case subset까지 만드는 단계라고 이해하면 됩니다.
