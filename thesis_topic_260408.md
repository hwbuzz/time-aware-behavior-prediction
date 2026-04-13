# 📄 논문 주제 및 연구 설계

## 1. 연구 주제
### SASRec 기반 행동 예측 모델에서 시간 정보 통합 효과 분석
- 부제 (선택): Time-aware Sequential Recommendation with Multi-task Learning

---

## 2. 연구 배경
실제 산업 환경에서 사용자 행동 시퀀스 분석은 다음 두 가지 방향으로 활용될 수 있다.

1. 다음 행동 예측 (What)
2. 다음 행동 + 발생 시점 예측 (What + When)

과거 A전자 프로젝트에서는 안정적인 구현을 위해 행동 예측 중심 모델(BERT4Rec)을 적용하였다.

하지만 실제 활용 측면에서는 “언제 발생하는지”까지 함께 예측하는 모델이 더 높은 가치를 가진다.

따라서 본 연구에서는 시간 정보를 포함한 행동 예측 모델의 효과를 정량적으로 검증하고자 한다.

---

## 3. 연구 목적
기존 sequential recommendation 모델은 행동 순서는 고려하지만 시간 간격(time gap)은 충분히 반영하지 않는다.

본 연구의 목적:
- 시간 정보를 입력으로 활용 시 성능 향상 여부 검증
- 시간 정보를 예측 대상으로 확장 시 What + When 동시 학습 가능성 확인
- multi-task learning 구조의 효과성 분석

---

## 4. 데이터셋
### BPI Challenge 2012 Event Log

- 금융기관 대출 신청 프로세스 이벤트 로그
- 주요 구성:
  - case id
  - activity
  - timestamp

특징:
- 프로세스 기반 sequence 구성 가능
- time delta 계산 가능
- Process Mining benchmark 데이터

---

## 5. 모델 설계
### Baseline: SASRec

- Transformer 기반 sequential model
- Unidirectional (causal)
- next item prediction에 적합
- 구조 단순 및 재현성 높음
- time prediction 확장 용이

---

## 6. 비교 실험 설계

### ① Base Model
- SASRec
- 입력: item sequence
- 출력: next item

### ② Time-aware Input Model
- SASRec + Time Embedding
- 입력:
  - item sequence
  - time delta
- 출력: next item

목적:
→ 시간 정보가 행동 예측 성능에 미치는 영향 분석

### ③ Multi-task Model
- SASRec + Time Embedding + Time Prediction Head
- 입력:
  - item sequence
  - time delta
- 출력:
  - next item
  - next time delta

목적:
→ What + When 동시 예측 성능 평가

---

## 7. 학습 구조

Loss:
L = L_item + λ * L_time

- item loss: Cross-Entropy
- time loss: MSE 또는 MAE

---

## 8. 평가 지표

### 행동 예측
- Hit@K
- NDCG@K

### 시간 예측
- MAE
- RMSE

---

## 9. 기대 결과

- 시간 정보 입력 시 성능 향상 여부 확인
- 시간 패턴 학습 가능성 검증
- What + When 동시 예측 모델의 실효성 확인

---

## 10. 모델 선택 근거

| 항목 | BERT4Rec | SASRec |
|------|----------|--------|
| 구조 | Bidirectional | Unidirectional |
| 목적 적합성 | Reconstruction | Next prediction |
| 구현 안정성 | 변동성 있음 | 안정적 |
| 시간 확장 | 복잡 | 용이 |

→ SASRec을 baseline으로 채택

---

## 🔥 한 줄 요약
SASRec 기반 모델에 시간 정보를 통합하여 행동(What)과 시간(When)을 동시에 예측하는 multi-task 구조의 효과를 검증하는 연구
