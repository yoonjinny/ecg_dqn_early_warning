# ECG Early Decision Making using DQN

### 강화학습 기반 조기 이상 탐지를 위한 동적 의사결정 모델 설계

> Dynamic Decision Making for Early Anomaly Detection using Deep Q-Network

---

## 프로젝트 개요

기존 ECG(심전도) 이상 탐지 연구는 주로 ECG 전체 신호를 관찰한 뒤 정상과 이상을 분류하는 데 초점을 맞추어 왔습니다. 최근에는 조기 경보(Early Warning)를 통해 보다 빠른 탐지를 시도하는 연구가 활발히 진행되고 있으나, 대부분 사전에 정의된 고정된 관찰 구간(Time Slice)을 기반으로 판단을 수행합니다.

그러나 실제 의사결정 환경에서는 모든 환자가 동일한 양의 정보를 필요로 하지 않습니다. 일부 ECG는 초기 구간의 정보만으로도 이상 여부를 판단할 수 있지만, 일부 ECG는 추가적인 관찰이 필요합니다.

본 연구는 이러한 문제의식에서 출발하여,

> **"무엇을 판단할 것인가"를 넘어 "언제 판단할 것인가"를 학습할 수 있는가?**

라는 질문을 강화학습 기반으로 검증하고자 하였습니다.

---

## 연구 목표

본 연구의 목적은 ECG 이상 탐지를 단순 분류 문제가 아닌 순차적 의사결정(Sequential Decision Making) 문제로 재정의하는 것입니다.

이를 위해 Agent는 ECG를 순차적으로 관찰하면서 다음 행동 중 하나를 선택하도록 설계하였습니다.

* Continue : 추가 관찰
* Predict Normal : 정상 판단
* Predict Abnormal : 이상 판단

즉, 기존 연구가 "무엇을 예측할 것인가"에 집중하였다면,

> **본 연구는 "언제 예측을 종료할 것인가"까지 함께 학습하는 정책을 설계하였습니다.**

---

## 데이터셋

### MIT-BIH Arrhythmia Dataset

* 총 샘플 수 : 109,554개
* ECG 길이 : 187 시점(Time Point)
* 원본 클래스 : 5개
* 최종 문제 정의 : 정상/비정상 이진 분류
* 출처 : https://www.kaggle.com/datasets/shayanfazeli/heartbeat

### 라벨 변환

| 원본 클래스 | 설명        | 최종 라벨        |
| ------ | --------- | ------------ |
| 0      | 정상        | Normal (0)   |
| 1      | 상심실성 조기수축 | Abnormal (1) |
| 2      | 심실성 조기수축  | Abnormal (1) |
| 3      | 융합 심박     | Abnormal (1) |
| 4      | 알 수 없음    | Abnormal (1) |

---

## 강화학습 환경 설계

### State

Agent가 현재까지 관찰한 정보

* ECG Signal
* Observed Ratio

Observed Ratio는 전체 ECG 중 현재까지 관찰한 비율을 의미합니다.

---

### Action

| Action | 의미               |
| ------ | ---------------- |
| 0      | Continue         |
| 1      | Predict Normal   |
| 2      | Predict Abnormal |

---

### Reward

| 상황               | Reward |
| ---------------- | -----: |
| Continue         |  -0.03 |
| Wrong Prediction |     -3 |
| Correct Normal   |     +1 |
| Correct Abnormal |     +6 |

Continue 행동에는 관찰 비용을 부여하였으며,

이상 탐지 실패에는 더 큰 패널티를 적용하여 정확도와 조기성 간의 균형을 학습하도록 설계하였습니다.

---

## DQN 구조

본 연구에서는 복잡한 특징 추출보다 판단 시점 학습에 초점을 두었습니다.

```text
Input State
(ECG Signal + Observed Ratio)
        ↓
      FC(128)
        ↓
       ReLU
        ↓
      FC(64)
        ↓
       ReLU
        ↓
Output (3 Actions)
```

---

## 실험 환경

| 항목                  | 설정           |
| ------------------- | ------------ |
| Framework           | PyTorch      |
| Environment         | Google Colab |
| GPU                 | Tesla T4     |
| Episode             | 3,000        |
| Batch Size          | 64           |
| Learning Rate       | 0.001        |
| Discount Factor (γ) | 0.99         |
| Seed                | 42, 77, 14   |

---

## 평가 지표

본 연구에서는 분류 성능뿐 아니라 조기 의사결정 효과를 함께 평가하였습니다.

* Accuracy
* Abnormal Precision
* Abnormal Recall
* Abnormal F1-score
* Average Observed Ratio

Average Observed Ratio는 전체 ECG 중 평균적으로 어느 정도의 정보를 관찰한 후 판단하였는지를 의미합니다.

---

# 실험 결과

## 1. 최종 DQN 성능

최종 Reward 설계를 적용한 DQN의 성능은 다음과 같습니다.

| Metric              | Score |
| ------------------- | ----: |
| Accuracy            | 0.897 |
| Abnormal Precision  |  0.72 |
| Abnormal Recall     |  0.67 |
| Abnormal F1-score   |  0.69 |
| Avg. Observed Ratio | 0.748 |

평균적으로 전체 ECG의 약 75%만 관찰하고도 안정적인 이상 탐지가 가능함을 확인하였습니다.

> 즉, 약 25%의 추가 정보를 읽지 않고도 일정 수준 이상의 의사결정이 가능함을 확인하였습니다.

---

## 2. 기존 분류 모델과의 비교

| Model               | Abnormal Precision | Abnormal Recall | Abnormal F1 | Observed Ratio |
| ------------------- | -----------------: | --------------: | ----------: | -------------: |
| Logistic Regression |               0.53 |            0.76 |        0.63 |           1.00 |
| LSTM                |               0.51 |            0.06 |        0.11 |           1.00 |
| MLP                 |               0.90 |            0.90 |        0.90 |           1.00 |
| DQN                 |               0.72 |            0.67 |        0.69 |           0.75 |

MLP는 가장 높은 분류 성능을 보였지만, 모든 샘플에 대해 전체 ECG를 관찰해야 했습니다.

반면 DQN은 일부 성능 손실을 감수하는 대신,

> **샘플마다 서로 다른 시점에서 판단을 종료하는 동적 의사결정이 가능함을 확인하였습니다.**

---

## 3. Seed 기반 재현성 평가

Seed 변경에도 Accuracy는 비교적 안정적으로 유지되었으며, 판단 시점과 비정상 탐지 성능에는 일부 변동이 존재하였습니다.

| Seed | Accuracy | Abnormal F1 | Avg. Observed Ratio |
| ---- | -------: | ----------: | ------------------: |
| 42   |    0.856 |       0.595 |               0.530 |
| 77   |    0.852 |       0.434 |               0.673 |
| 14   |    0.863 |       0.571 |               0.506 |

### Mean ± Std

| Metric              | Mean ± Std    |
| ------------------- | ------------- |
| Accuracy            | 0.857 ± 0.005 |
| Abnormal F1         | 0.533 ± 0.072 |
| Avg. Observed Ratio | 0.570 ± 0.074 |

Seed 변화에 따른 실험을 통해 정책의 일반화 가능성과 안정성을 함께 평가하였습니다.

---

## 연구의 의의

본 연구는 ECG 조기 이상 탐지 문제를 강화학습 기반의 순차적 의사결정 문제로 재정의하였습니다.

기존 연구가 "무엇을 판단할 것인가"에 집중하였다면,

> **본 연구는 "언제 판단할 것인가"를 학습 대상으로 정의하였다는 점에서 차별성을 가집니다.**

최고 수준의 분류 성능을 달성한 연구는 아니지만,

제한된 정보만으로 조기 의사결정을 수행할 수 있는 가능성을 확인하였으며, 향후 AI Agent 기반의 동적 의사결정 환경으로 확장될 수 있는 가능성을 제시하였습니다.

---

## 한계점 및 향후 연구

* 조기성과 정확도 간 Trade-off가 존재함을 확인하였습니다.
* Reward 설계 및 초기 탐색 과정에 따라 정책 편차가 발생하였습니다.
* 본 연구에 사용된 ECG 데이터는 187 시점의 비교적 짧은 시계열 데이터로, 판단 시점 최적화 효과를 충분히 검증하기에는 한계가 존재합니다.
* 향후 장기 시계열 데이터 기반 검증이 필요합니다.
* 네트워크 이상탐지, 예지보전, 실시간 모니터링 등 동적 조기 의사결정이 중요한 다양한 산업 분야로의 확장 가능성이 기대됩니다.

---

## 핵심 기여 (Contributions)

* ECG 조기 이상 탐지를 순차적 의사결정 문제로 재정의
* DQN을 활용하여 판단 시점까지 함께 학습하는 정책 설계
* 기존 분류 모델 대비 약 25% 적은 정보만으로 의사결정 가능성 검증
* Seed 기반 재현성 평가를 통한 일반화 성능 분석
* 다양한 동적 의사결정 문제로의 확장 가능성 제시
