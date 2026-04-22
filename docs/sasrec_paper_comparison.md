# SASRec 논문 성능과 BPI2012 전체 데이터 실험 비교

## 참고 논문

Kang & McAuley, **Self-Attentive Sequential Recommendation**, ICDM 2018.

- 논문 PDF: <https://cseweb.ucsd.edu/~jmcauley/pdfs/icdm18.pdf>

## 논문 평가 설정 요약

SASRec 원 논문은 user별 sequence에서 마지막 item을 test, 그 직전 item을 validation, 나머지를 train으로 나누는 leave-one-out 방식을 사용한다. 평가 시에는 각 user마다 ground-truth item 1개와 random negative item 100개를 함께 rank한 뒤 Hit@10과 NDCG@10을 계산한다.

우리 BPI2012 실험도 큰 틀에서는 같은 방식이다.

- 마지막 event: test
- 그 직전 event: validation
- 나머지 event: train
- negative samples: 100
- metrics: HR@10, NDCG@10
- eval users: 전체 user 평가 (`eval_users=0`)

논문 기본 hyperparameter는 대체로 다음과 같다.

- learning rate: 0.001
- batch size: 128
- self-attention blocks: 2
- learned positional embedding 사용
- shared item embedding 사용
- max sequence length: Amazon/Steam은 50, ML-1M은 200

## SASRec 논문 성능

논문 Table III 기준 SASRec 성능은 다음과 같다.

| Dataset | Hit@10 | NDCG@10 |
|---|---:|---:|
| Amazon Beauty | 0.4854 | 0.3219 |
| Amazon Games | 0.7410 | 0.5360 |
| Steam | 0.8729 | 0.6306 |
| MovieLens-1M | 0.8245 | 0.5905 |

## 우리 BPI2012 전체 데이터 실험 성능

전체 데이터는 `data/processed/bpi2012_complete_only/sasrec_interactions.txt`를 사용했다.

데이터 규모:

| 항목 | 값 |
|---|---:|
| Users | 13,087 |
| Items | 23 |
| Train interactions | 138,332 |
| Full SASRec rows | 164,506 |

학습 설정:

| Parameter | Value |
|---|---:|
| batch_size | 128 |
| lr | 0.001 |
| maxlen | 50 |
| hidden_units | 50 |
| num_blocks | 2 |
| num_heads | 1 |
| dropout_rate | 0.2 |
| num_epochs | 50 |
| eval_every | 5 |
| eval_users | 0 |
| num_negative_samples | 100 |
| topk | 10 |
| seed | 42 |
| device | cpu |

결과:

| Checkpoint | Epoch | Valid NDCG@10 | Valid HR@10 | Test NDCG@10 | Test HR@10 |
|---|---:|---:|---:|---:|---:|
| Best validation | 15 | 0.5428 | 0.5735 | 0.4071 | 0.4412 |
| Last epoch | 50 | 0.5036 | 0.5366 | 0.4503 | 0.4880 |

저장된 best checkpoint를 다시 로드해서 inference-only 평가를 수행한 결과:

| Reload checkpoint | Valid NDCG@10 | Valid HR@10 | Test NDCG@10 | Test HR@10 |
|---|---:|---:|---:|---:|
| `sasrec_best.pth` | 0.5428 | 0.5734 | 0.4074 | 0.4419 |

대표 성능은 test 성능을 보고 고르는 것이 아니라 validation 기준으로 선택하는 것이 맞으므로, **Best validation checkpoint의 Test NDCG@10 = 0.4071, Test HR@10 = 0.4412**를 대표값으로 보는 것이 적절하다.

## 논문 성능과 우리 성능 비교

| 비교 대상 | 논문 Hit@10 | 논문 NDCG@10 | 우리 Test HR@10 | 우리 Test NDCG@10 |
|---|---:|---:|---:|---:|
| Amazon Beauty | 0.4854 | 0.3219 | 0.4412 | 0.4071 |
| Amazon Games | 0.7410 | 0.5360 | 0.4412 | 0.4071 |
| Steam | 0.8729 | 0.6306 | 0.4412 | 0.4071 |
| MovieLens-1M | 0.8245 | 0.5905 | 0.4412 | 0.4071 |

해석:

- 우리 BPI2012 실험의 NDCG@10은 Amazon Beauty보다 높다.
- 우리 BPI2012 실험의 Hit@10은 Amazon Beauty보다 약간 낮다.
- Amazon Games, Steam, MovieLens-1M과 비교하면 HR@10과 NDCG@10 모두 낮다.
- 따라서 현재 결과는 논문 benchmark 중 가장 sparse한 Beauty와 비교하면 NDCG는 더 좋지만 HR은 낮고, 더 dense하거나 추천 신호가 강한 dataset들보다는 낮은 수준이다.

## 주의할 점

이 비교는 직접적인 우열 비교라기보다 참고 비교에 가깝다.

1. **데이터 도메인이 다르다.**

   논문은 product/movie/game recommendation dataset을 사용하고, 우리는 BPI2012 process event next-activity prediction을 수행한다. user와 item의 의미가 다르며 sequence가 생성되는 메커니즘도 다르다.

2. **우리 item 수가 매우 작다.**

   BPI2012 COMPLETE-only setting에서는 item 수가 23개뿐이다. 논문 benchmark는 item 수가 수천에서 수만 개 수준이다. 따라서 100 negative sampling 평가가 논문과 완전히 같은 난이도를 갖는다고 보기는 어렵다.

3. **현재 평가의 negative sampling 방식은 작은 item vocabulary에서 왜곡될 수 있다.**

   item 수가 23개인데 negative sample을 100개 뽑으면 중복 negative가 발생할 수 있다. 논문 데이터처럼 item 수가 매우 큰 경우에는 이 문제가 거의 드러나지 않지만, BPI2012처럼 item vocabulary가 작은 경우에는 평가값 해석에 주의가 필요하다.

4. **Last epoch test 성능이 더 높더라도 대표 성능으로 쓰기는 조심스럽다.**

   epoch 50의 Test NDCG@10은 0.4503으로 더 높지만, test 성능을 보고 checkpoint를 선택하면 test leakage 성격이 생긴다. 논문 방식에 맞추려면 validation 성능이 가장 높은 epoch 15 checkpoint를 대표 모델로 보는 것이 더 타당하다.

## 결론

현재 전체 BPI2012 COMPLETE-only 데이터에서 학습한 SASRec은 validation 기준 best checkpoint에서 다음 성능을 보였다.

- Test HR@10: **0.4412**
- Test NDCG@10: **0.4071**

SASRec 원 논문 성능과 비교하면, 우리 결과는 Amazon Beauty보다 NDCG@10은 높지만 HR@10은 조금 낮고, Amazon Games, Steam, MovieLens-1M보다는 낮다. 다만 dataset과 task 성격이 다르기 때문에 논문 benchmark 수치와의 직접 비교보다는, 같은 BPI2012 split에서 PopRec, Markov Chain, GRU4Rec, Caser 등 baseline을 동일한 metric으로 평가해 비교하는 것이 더 설득력 있는 성능 분석이 될 것이다.

## 관련 저장 파일

학습/평가 결과는 다음 위치에 저장되어 있다.

- 결과 확인 노트북: `notebooks/sasrec_full_data_results.ipynb`
- 실험 비교 index: `outputs/sasrec_bpi2012_full/experiment_index.csv`
- 학습 run summary/history/checkpoints: `outputs/sasrec_bpi2012_full/full_default_50ep_seed42/`
- best checkpoint reload 평가 summary: `outputs/sasrec_bpi2012_full/full_default_50ep_seed42_best_reload_eval/`

