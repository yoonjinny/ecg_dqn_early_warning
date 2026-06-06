# DQN 기반 ECG 조기 이상 경보 시스템

## 1. Project Goal

이 프로젝트는 ECG heartbeat 시계열을 순차적으로 관찰하면서, DQN agent가 다음 행동을 선택하도록 설계한다.

- `Continue`: ECG를 더 관찰한다.
- `Predict Normal`: 정상으로 판단하고 episode를 종료한다.
- `Predict Abnormal`: 이상으로 판단하고 조기 경보를 발생시킨다.

목표는 단순히 정확한 분류를 하는 것이 아니라, 가능한 적은 관찰 길이로 정상/이상을 판단하는 것이다.

## 2. Dataset

권장 데이터셋은 Kaggle ECG Heartbeat Categorization Dataset이다.

- Kaggle dataset: `shayanfazeli/heartbeat`
- 파일 예시: `mitbih_train.csv`, `mitbih_test.csv`
- 각 row는 길이 187 ECG signal + label로 구성된다.
- label 0은 normal, label 1~4는 abnormal로 binary 변환한다.

데이터를 넣지 않으면 demo synthetic ECG 데이터로 코드가 실행된다.

## 3. MDP Design

### State

DQN 입력은 고정 길이 vector다.

```text
state = [observed ECG values, zero padding, observed_ratio]
```

예를 들어 187 길이 ECG에서 30개만 관찰했다면:

```text
[x1, x2, ..., x30, 0, 0, ..., 0, 30/187]
```

### Action

```text
0: Continue
1: Predict Normal
2: Predict Abnormal
```

### Reward

```text
Correct normal prediction   = 1.0 - observed_ratio
Correct abnormal prediction = 3.0 - observed_ratio
Wrong prediction            = -2.0
Continue                    = -0.01
```

이 reward는 이상 경보 recall과 조기 판단을 더 중요하게 반영한다.

## 4. How to Run

```bash
pip install -r requirements.txt
cd src
python train.py
python evaluate.py
python baseline.py
```

## 5. Experiments

권장 실험:

1. step size 변경: 5, 10, 20
2. reward 변경: abnormal reward 2.0, 3.0, 5.0
3. epsilon decay 변경
4. random seed 3회 이상 반복

## 6. Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Average Observed Ratio

핵심 비교:

```text
Baseline classifier: ECG 100% 관찰 후 분류
DQN agent: ECG 일부만 관찰 후 조기 판단
```

## 7. Report Structure

1. 프로젝트 주제 및 목표
2. 환경 및 데이터셋 설명
3. State, Action, Reward 설계
4. DQN 알고리즘 및 hyperparameter
5. 실험 셋업 및 평가 지표
6. 실험 결과
7. 토의 및 결론
