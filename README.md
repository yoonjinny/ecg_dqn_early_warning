# ECG Early Decision Making using DQN

## 프로젝트 개요

기존 ECG 이상 탐지 연구는 대부분 ECG 전체 데이터를 관찰한 후 정상과 이상을 분류하는 데 초점을 맞추어 왔습니다. 최근에는 조기 경보(Early Warning)에 대한 연구가 활발히 진행되고 있지만, 여전히 고정된 관찰 구간(Time Slice)을 기반으로 판단하는 경우가 많습니다.

그러나 실제 환경에서는 모든 환자가 동일한 양의 정보를 필요로 하지 않습니다. 일부 ECG는 초기 구간의 정보만으로도 이상 여부를 판단할 수 있지만, 일부 ECG는 추가적인 관찰이 필요합니다. 따라서 "무엇을 판단할 것인가"뿐만 아니라 "언제 판단할 것인가" 역시 중요한 문제라고 볼 수 있습니다.

본 프로젝트에서는 ECG 조기 의사결정 문제를 강화학습 환경으로 재정의하고, DQN(Deep Q-Network)을 활용하여 제한된 정보만으로도 최적의 판단 시점을 학습할 수 있는지를 검증하였습니다.

---

## 연구 목표

* ECG 조기 의사결정 문제를 강화학습 환경으로 정의
* 정상/비정상 분류뿐 아니라 판단 시점까지 학습
* 제한된 정보만으로 이상 여부를 조기에 판단할 수 있는 정책 학습
* 기존 분류 모델과의 성능 및 관찰 길이 비교

---

## 데이터셋

### MIT-BIH Arrhythmia Dataset

* 총 109,554개 ECG 샘플
* 각 샘플은 187개의 시계열 포인트로 구성
* 5개 클래스를 이진 분류로 변환
* 데이터 출처 : https://www.kaggle.com/datasets/shayanfazeli/heartbeat

| 원본 클래스 | 설명        |     개수 |
| ------ | --------- | -----: |
| 0      | 정상        | 72,471 |
| 1      | 상심실성 조기수축 |  2,223 |
| 2      | 심실성 조기수축  |  5,788 |
| 3      | 융합 심박     |    641 |
| 4      | 알 수 없음    |  6,431 |

### 라벨 변환

* 정상(0) → 0
* 나머지 클래스(1~4) → 1 (비정상)

---

## 데이터 전처리

* Train/Test Split (8:2)
* Stratified Sampling 적용
* Random Seed 고정
* 이진 분류 라벨 변환

---

## 강화학습 환경 설계

### State

* 현재까지 관찰한 ECG 신호
* Observed Ratio

  * 전체 ECG 중 현재까지 관찰한 비율

### Action

| Action | 의미               |
| ------ | ---------------- |
| 0      | Continue (추가 관찰) |
| 1      | Predict Normal   |
| 2      | Predict Abnormal |

### Reward

| 상황               | Reward |
| ---------------- | -----: |
| Continue         |  -0.03 |
| Wrong Prediction |     -3 |
| Correct Normal   |     +1 |
| Correct Abnormal |     +6 |

---

## DQN 구조

### Q-Network

```
Input State (ECG + Observed Ratio)
        ↓
      FC(128)
        ↓
       ReLU
        ↓
       FC(64)
        ↓
       ReLU
        ↓
    Output (3)
```

### 학습 과정

```
State 관찰
    ↓
ε-greedy Action 선택
    ↓
Reward 획득
    ↓
Replay Buffer 저장
    ↓
Mini-batch 학습
    ↓
Target Network 동기화
```

---

## 실험 환경

| 항목            | 설정           |
| ------------- | ------------ |
| Framework     | PyTorch      |
| Environment   | Google Colab |
| GPU           | Tesla T4     |
| Episodes      | 3,000        |
| Batch Size    | 64           |
| Gamma         | 0.99         |
| Learning Rate | 1e-3         |
| Seed          | 42, 77, 14   |

---

## 평가 지표

* Accuracy
* Abnormal Precision
* Abnormal Recall
* Abnormal F1-score
* Average Observed Ratio

Average Observed Ratio는 전체 ECG 중 평균적으로 어느 정도의 정보를 관찰한 후 판단하였는지를 의미합니다.

---

## 실험 결과

### DQN (Reward E3)

| Metric              |    성능 |
| ------------------- | ----: |
| Accuracy            | 0.897 |
| Abnormal Precision  |  0.72 |
| Abnormal Recall     |  0.67 |
| Abnormal F1-score   |  0.69 |
| Avg. Observed Ratio | 0.748 |

평균적으로 전체 ECG의 약 75%만 관찰하고도 안정적인 이상 탐지가 가능함을 확인하였습니다.

---

### 기존 모델 비교

| 모델                  | Abnormal Precision | Abnormal Recall | Abnormal F1 | Observed Ratio |
| ------------------- | -----------------: | --------------: | ----------: | -------------: |
| Logistic Regression |               0.53 |            0.76 |        0.63 |           1.00 |
| LSTM                |               0.51 |            0.06 |        0.11 |           1.00 |
| MLP                 |               0.90 |            0.90 |        0.90 |           1.00 |
| DQN (E3)            |               0.72 |            0.67 |        0.69 |           0.75 |

DQN은 최고 수준의 분류 성능을 달성하지는 못하였지만, 기존 모델보다 적은 정보만으로 의사결정을 수행할 수 있다는 장점을 확인하였습니다.

---

### Seed 변경에 따른 일반화 성능

| Seed | Accuracy | Abnormal F1 | Avg. Observed Ratio |
| ---- | -------: | ----------: | ------------------: |
| 42   |    0.856 |       0.595 |               0.530 |
| 77   |    0.852 |       0.434 |               0.673 |
| 14   |    0.863 |       0.571 |               0.506 |

#### Mean ± Std

| Metric              | Mean ± Std    |
| ------------------- | ------------- |
| Accuracy            | 0.857 ± 0.005 |
| Abnormal F1         | 0.533 ± 0.072 |
| Avg. Observed Ratio | 0.570 ± 0.074 |

---

## 결론

본 연구는 ECG 조기 의사결정 문제를 강화학습 기반 문제로 재정의하고, 정상/비정상 분류뿐 아니라 판단 시점까지 학습할 수 있음을 확인하였습니다.

비록 최고 수준의 분류 성능을 달성하지는 못하였지만, 제한된 정보만으로 조기 의사결정을 수행할 수 있는 가능성을 확인하였으며, 향후 동적인 의사결정이 요구되는 다양한 분야로의 확장 가능성을 제시하였습니다.

---

## 한계점 및 향후 연구

* 조기성과 정확도 간 Trade-off 존재
* Reward 설계에 따른 정책 변동 발생
* ECG 데이터의 짧은 관찰 구간으로 인해 판단 시점 최적화 효과를 충분히 검증하기 어려움
* 장기 시계열 데이터 기반 검증 필요
* 네트워크 이상탐지, 예지보전, 실시간 모니터링 등 동적 의사결정 문제로 확장 가능

---

## 핵심 기여

> 기존 ECG 연구가 "무엇을 판단할 것인가"에 집중하였다면, 본 연구는 "언제 판단할 것인가"를 학습 대상으로 정의하고 조기 의사결정의 가능성을 검증하였습니다.
