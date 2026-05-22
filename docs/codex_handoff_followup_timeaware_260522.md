# Codex Handoff: SASRec Time-Aware Follow-up (2026-05-22)

## 1. Project goal

This project studies next-activity prediction on the BPI 2012 complete-only dataset using SASRec.

Current research flow:

1. Implement and verify SASRec baseline
2. Add time-aware variants and compare against the strongest baseline
3. Decide whether time-aware gives a meaningful gain
4. If needed, continue with follow-up experiments

The user wants a future Codex session on another computer to continue from the current state without re-deriving all prior context.


## 2. Dataset / task context

- Dataset: BPI 2012 complete-only
- Task: next activity prediction
- Important characteristic: item/activity vocabulary is small
  - This is one reason why full ranking evaluation is considered the main evaluation.

Main evaluation convention used so far:

- Main metric: full ranking + NDCG@10
- Secondary: full ranking + NDCG@5
- Supplementary: negative sampling(100) + NDCG@10 / NDCG@5


## 3. Important code status

Key files:

- `C:\Users\hyewoo choi\Documents\99. 대학원\0. 논문\git\time-aware-behavior-prediction\src\train_sasrec.py`
- `C:\Users\hyewoo choi\Documents\99. 대학원\0. 논문\git\time-aware-behavior-prediction\src\sasrec_utils.py`
- `C:\Users\hyewoo choi\Documents\99. 대학원\0. 논문\git\time-aware-behavior-prediction\src\sasrec_model.py`

Current codebase supports all of the following:

- baseline SASRec
- time-aware additive + bucket
- time-aware additive + continuous
- time-aware attention bias

Important historical fixes already applied:

1. Full ranking evaluation bug fixed
2. Sampled test protocol aligned more carefully
3. Additive continuous path implemented
4. Attention bias path implemented
5. Attention bias bug fixed so raw time values are actually passed through
6. Metadata fields for time-aware runs cleaned up

Attention bias is available through the current code and is not just a notebook-level hack.


## 4. Time-aware variants implemented so far

### A. Additive

Input representation:

- `item embedding + position embedding + time embedding`

Two time encodings were implemented:

1. Bucket
   - time gap -> bucket id -> embedding lookup
2. Continuous
   - `log1p(time gap)` -> linear projection -> hidden-size vector

Time sources used:

1. `delta_prev_seconds`
   - gap from previous event
2. `delta_start_seconds`
   - elapsed time since case start


### B. Attention bias

Instead of adding time to the input embedding, time is added to the attention score:

- base attention score + time-based bias

Implemented design:

- time source: `delta_start_seconds`
- pairwise causal gap between events
- bucketized pairwise gap
- bucket-specific scalar bias added to self-attention score

This is the most promising time-aware family so far.


## 5. Bucket definitions used

### b8

Boundaries:

- `60, 600, 3600, 86400`

Meaning:

- first event
- zero gap
- (0, 1 minute)
- [1 minute, 10 minutes)
- [10 minutes, 1 hour)
- [1 hour, 1 day)
- [1 day, +)


### b9

Boundaries:

- `60, 600, 3600, 86400, 604800`

Meaning:

- first event
- zero gap
- (0, 1 minute)
- [1 minute, 10 minutes)
- [10 minutes, 1 hour)
- [1 hour, 1 day)
- [1 day, 7 days)
- [7 days, +)


### b10

Boundaries:

- `10, 60, 600, 3600, 86400, 604800`

Meaning:

- first event
- zero gap
- (0, 10 seconds)
- [10 seconds, 1 minute)
- [1 minute, 10 minutes)
- [10 minutes, 1 hour)
- [1 hour, 1 day)
- [1 day, 7 days)
- [7 days, +)


## 6. Baseline conclusion

Baseline rerun and sanity check were done after evaluation fixes.

Relevant notebooks:

- `C:\Users\hyewoo choi\Documents\99. 대학원\0. 논문\git\time-aware-behavior-prediction\notebooks\sasrec_bpi2012_colab_train_evalfix_260512.ipynb`
- `C:\Users\hyewoo choi\Documents\99. 대학원\0. 논문\git\time-aware-behavior-prediction\notebooks\sasrec_bpi2012_colab_train_sanity_check_evalfix_260513.ipynb`

Final baseline choice:

- `anchor_ml20`

Main reason:

- strongest overall performance
- reasonably stable across seeds

Best competing baseline:

- `refine_ml50_do035`


## 7. Time-aware experiment notebooks and what they mean

Reference summary file:

- `C:\Users\hyewoo choi\Documents\99. 대학원\0. 논문\git\time-aware-behavior-prediction\docs\notebook_summary_baseline_timeaware_260514.md`

Main time-aware notebooks:

### 02

- File: `C:\Users\hyewoo choi\Documents\99. 대학원\0. 논문\git\time-aware-behavior-prediction\notebooks\sasrec_timeaware_bpi2012_colab_train_02_260513.ipynb`
- Purpose:
  - apply additive bucket time-aware to `anchor_ml20`
- Conclusion:
  - baseline is better than additive time-aware

### 03

- File: `C:\Users\hyewoo choi\Documents\99. 대학원\0. 논문\git\time-aware-behavior-prediction\notebooks\sasrec_timeaware_bpi2012_colab_train_03_260513.ipynb`
- Purpose:
  - apply additive bucket time-aware to `refine_ml50_do035`
- Conclusion:
  - some validation fit signal, but no stable test improvement over baseline

### 04

- File: `C:\Users\hyewoo choi\Documents\99. 대학원\0. 논문\git\time-aware-behavior-prediction\notebooks\sasrec_timeaware_bpi2012_colab_train_04_260514.ipynb`
- Purpose:
  - apply additive continuous time-aware to `refine_ml50_do035`
- Conclusion:
  - continuous additive was not clearly better than bucket additive

### 05

- File: `C:\Users\hyewoo choi\Documents\99. 대학원\0. 논문\git\time-aware-behavior-prediction\notebooks\sasrec_timeaware_bpi2012_colab_train_05_260514.ipynb`
- Purpose:
  - compare `delta_start` additive bucket vs `delta_start` additive continuous on `refine_ml50_do035`
- Conclusion:
  - bucket was better than continuous, but still not clearly better than baseline

### 06

- File: `C:\Users\hyewoo choi\Documents\99. 대학원\0. 논문\git\time-aware-behavior-prediction\notebooks\sasrec_timeaware_bpi2012_colab_train_06_260514.ipynb`
- Purpose:
  - apply `delta_start + attention bias + b9` to `refine_ml50_do035`
- Conclusion:
  - best time-aware result so far

### 07

- File: `C:\Users\hyewoo choi\Documents\99. 대학원\0. 논문\git\time-aware-behavior-prediction\notebooks\sasrec_timeaware_bpi2012_colab_train_07_260514.ipynb`
- Purpose:
  - apply `delta_start + attention bias + b9` to `anchor_ml20`
- Conclusion:
  - better than additive time-aware, but still worse than `anchor_ml20` baseline

### 08

- File: `C:\Users\hyewoo choi\Documents\99. 대학원\0. 논문\git\time-aware-behavior-prediction\notebooks\sasrec_timeaware_bpi2012_colab_train_08_260515.ipynb`
- Purpose:
  - refine attention bias bucket boundaries (`b10`) on `refine_ml50_do035`
- Conclusion:
  - `b10` did not beat the earlier `b9`


## 8. Current best conclusions

### Overall strongest model

- `anchor_ml20` baseline

### Best time-aware candidate

- `refine_ml50_do035 + delta_start_seconds + attention bias + b9`

### Overall interpretation

- Time-aware is not completely useless.
- Additive time-aware did not consistently improve over the strongest baseline.
- Attention bias is consistently more promising than additive methods.
- Even so, no time-aware method has yet beaten the strongest baseline `anchor_ml20` on the main metric in a convincing way.


## 9. Current practical ranking

Main metric: full ranking + NDCG@10

### For anchor_ml20 family

- best: baseline
- next: attention bias
- then: additive bucket b9
- then: additive bucket b8

### For refine_ml50_do035 family

- best overall within that family: baseline or very close attention bias b9 depending on metric view
- best time-aware within that family: attention bias b9
- then: attention bias b10
- additive variants below that


## 10. Important interpretation for future work

The key pattern from all current experiments is:

- changing the time source alone did not solve the problem
- changing the way time is injected into the model mattered more
- attention bias worked better than additive time embedding

This suggests:

- "where time is used" may matter more than "how time is bucketized"


## 11. Existing result summary artifacts

Useful files:

- notebook summary
  - `C:\Users\hyewoo choi\Documents\99. 대학원\0. 논문\git\time-aware-behavior-prediction\docs\notebook_summary_baseline_timeaware_260514.md`
- combined result workbook from notebooks 02~08
  - `C:\Users\hyewoo choi\Documents\99. 대학원\0. 논문\git\time-aware-behavior-prediction\docs\timeaware_results_02_08_260517.xlsx`
- formatted workbook based on professor-facing template
  - `C:\Users\hyewoo choi\Documents\99. 대학원\0. 논문\git\time-aware-behavior-prediction\docs\모델성능_02_08_정리_260517.xlsx`


## 12. What the user may ask next

Likely next-step requests:

1. Interpret a new follow-up experiment result
2. Decide whether more time-aware experiments are worth doing
3. Prepare professor-facing summary text
4. Prepare report/thesis tables
5. Compare time-aware approaches under the main metric only


## 13. Good default recommendations for the next Codex

If the user asks whether more time-aware experiments are needed:

- say that broad exploration is already substantial
- if doing one more experiment, refine attention bias rather than reopening additive variants
- otherwise move toward writing / interpretation

If the user asks for the most important single takeaway:

- overall winner is still `anchor_ml20` baseline
- best time-aware candidate is `refine_ml50_do035 + delta_start + attention bias + b9`

If the user asks about the main comparison basis:

- use full ranking + NDCG@10 as the main metric


## 14. Notebook execution preference

The user explicitly prefers that Colab notebooks include the following as active code, not commented out:

```python
%cd /content/time-aware-behavior-prediction
!git pull
```

This should be included for future Colab notebooks.


## 15. Suggested short briefing sentence for a new Codex session

If a new Codex session needs a one-paragraph briefing, use something like this:

"This repo contains SASRec baseline and time-aware extensions for BPI 2012 complete-only next-activity prediction. The strongest overall model so far is the anchor_ml20 baseline. Additive time-aware variants did not consistently beat it. The most promising time-aware setup was refine_ml50_do035 + delta_start_seconds + 9-bucket attention bias, but it still did not clearly beat anchor_ml20 on the main metric (full ranking + NDCG@10). Existing time-aware notebooks are 02~08, with 06 being the best time-aware result and 08 showing that a b10 attention-bias refinement did not beat b9."
