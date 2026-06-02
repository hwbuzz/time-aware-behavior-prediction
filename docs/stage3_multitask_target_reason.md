# Process Mining / Predictive Process Monitoring에서 Time Prediction Target: Remaining Time vs Next-event Time

짧게 말하면,

**Predictive Process Monitoring(PPM) / Process Mining 문헌에서는 `remaining time prediction`이 더 보편적이고 사실상 표준(task benchmark)에 가깝고, `next-event timestamp` 또는 `delta-to-next-event prediction`은 상대적으로 덜 보편적이지만 점점 늘어나는 세부 태스크**이다.

다만 연구 목적에 따라 의미가 꽤 달라진다.

---

## 1. Remaining Time Prediction — 가장 전통적이고 표준적인 문제

### 정의

현재 시점(prefix) 이후 프로세스 종료(case completion)까지 남은 시간을 예측하는 문제

예시:

* 보험 청구 프로세스가 끝날 때까지 몇 일이 남았는가?
* 주문 처리 완료까지 얼마나 걸리는가?

### 왜 표준이 되었는가?

* 실무적 가치가 큼 (SLA 관리, 지연 예측, 리소스 계획)
* 공개 benchmark와 초기 연구들이 많이 사용
* Process Mining community에서 “time prediction”이라고 하면 기본적으로 remaining time을 떠올리는 경우가 많음

### 입력 / 출력 예시

입력:
[A, B, C] + timestamps

출력:
remaining_time = 3.4 days

### 특징

* 회귀(regression) 문제
* prefix마다 prediction 수행
* 평가 지표: MAE, RMSE, MAPE 등

---

## 2. Next-event Timestamp / Delta-to-next-event Prediction — 더 세밀한(Event-level) 예측

### 정의

다음 이벤트가 언제 발생할지를 예측하는 문제

대표적으로 두 가지 방식이 존재한다.

### (1) Absolute Timestamp Prediction

예:
next_timestamp = 2026-06-10 13:45

### (2) Delta-to-next-event Prediction (더 일반적)

예:
time_to_next_event = 5.3 hours

실제로는 absolute timestamp보다 **delta(time gap)** 방식이 더 많이 사용된다.

### 이유

* 모델링이 더 쉬움
* 분포가 상대적으로 안정적
* normalization이 용이함
* sequence model과 결합하기 쉬움

### 입력 / 출력 예시

입력:
A(t1), B(t2), C(t3)

출력:
Δt_next = t4 − t3

### 특징

* event-level local prediction
* sequential modeling과 궁합이 좋음
* next activity prediction과 multi-task learning 구성이 쉬움

---

## 3. SASRec / Transformer 기반 Sequential Prediction에서는?

여기서 중요한 포인트가 있다.

SASRec 기반 sequential prediction 구조에서는 **remaining time prediction보다 next-event delta prediction이 훨씬 자연스럽다.**

SASRec 자체는 다음과 같은 구조를 가진다.

x1, x2, x3 → x4

즉, 다음 행동(next event)을 예측하는 구조이다.

시간 예측을 함께 수행하면 다음과 같이 자연스럽게 확장할 수 있다.

x1, x2, x3 → x4
Δt4

반면 remaining time prediction은 아래와 같은 구조가 된다.

x1, x2, x3 → case_end_remaining_time

이는 sequential recommendation 구조와 약간 결이 다르며, 다음 행동 예측과의 결합도 상대적으로 덜 자연스럽다.

---

## 4. 비교 정리

| 관점                        | Remaining Time | Next-event Delta |
| ------------------------- | -------------- | ---------------- |
| Process Mining 전통         | 매우 높음          | 상대적으로 낮음         |
| Sequential Modeling 적합성   | 낮음             | 매우 높음            |
| SASRec와 자연스러운 결합          | 낮음             | 매우 높음            |
| Next Activity와 Multi-task | 애매             | 매우 쉬움            |
| 구현 난이도                    | 중간             | 쉬움               |

---

## 5. 논문/캡스톤에서 설명할 수 있는 논리

다음과 같이 연구 설계 논리를 설명할 수 있다.

“Process Mining 분야에서는 remaining time prediction이 대표적인 시간 예측 문제로 널리 연구되어 왔다. 그러나 본 연구는 Transformer 기반 sequential prediction 관점에서 사용자 행동 시퀀스의 다음 행동(next event) 예측과의 결합 가능성을 고려하여, 프로세스 종료까지의 잔여시간이 아닌 다음 행동까지의 시간 간격(time-to-next-event, delta-to-next-event)을 예측 대상으로 설정한다.”

이와 같은 설명은 “왜 remaining time prediction이 아니라 next-event time prediction을 선택했는가?”에 대한 연구적 정당성을 제공한다.
