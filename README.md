# 강화학습 기반 조기 이상 탐지를 위한 동적 의사결정 모델 설계

### ECG 심전도 데이터를 활용한 Early Decision Making DQN 모델

> Dynamic Decision Making for Early Anomaly Detection using Deep Q-Network

---

## 프로젝트 개요

기존 ECG 이상 탐지 연구는 크게 두 가지 방향으로 발전해 왔습니다.

첫째는 **어떤 데이터를 이상으로 분류할 것인가**에 대한 문제입니다. CNN, LSTM 등의 분류 모델은 ECG 전체 신호를 관찰한 뒤 정상과 이상을 구분하는 데 집중하였습니다.

둘째는 **얼마나 빨리 탐지할 것인가**에 대한 문제입니다. 조기 경보(Early Warning) 연구에서는 일부 구간만 활용하여 빠른 탐지를 시도하였지만, 대부분 사람이 사전에 정의한 고정된 관찰 길이(Time Slice)를 사용하였습니다.

그러나 실제 의사결정 환경에서는 또 다른 질문이 존재합니다.

> **언제 판단하는 것이 가장 적절한가?**

모든 ECG가 동일한 길이의 관찰을 필요로 하지는 않습니다.

어떤 ECG는 초기 구간만으로도 이상 여부를 판단할 수 있지만, 어떤 ECG는 추가적인 관찰이 필요합니다.

본 프로젝트는 이러한 문제의식에서 출발하여,

**판단 시점 자체를 학습할 수 있을까?**

라는 질문을 강화학습으로 검증하고자 하였습니다.

---

## 연구 목표

본 연구의 목적은 ECG 조기 이상 탐지를 단순 분류 문제가 아닌,

> **동적 의사결정(Sequential Decision Making) 문제로 재정의하는 것**

입니다.

이를 위해 Agent는 ECG를 순차적으로 관찰하면서 다음 행동 중 하나를 선택합니다.

* Continue : 추가 관찰
* Predict Normal : 정상 판단
* Predict Abnormal : 이상 판단

즉,

기존 모델이 "무엇을 예측할 것인가"를 학습했다면,

본 연구에서는

> **"언제 예측을 종료할 것인가"까지 함께 학습하는 정책(Policy)을 설계하였습니다.**

---

## 데이터셋

### MIT-BIH Arrhythmia Dataset

* 제공기관 : MIT & Beth Israel Hospital
* ECG 기록 수 : 48개
* 샘플링 주파수 : 360Hz
* 데이터 길이 : 약 30분
* 데이터 형태 : 2채널 ECG
* 데이터 출처 : https://www.kaggle.com/datasets/shayanfazeli/heartbeat
  

각 샘플은 총 187개의 ECG 시계열 포인트로 구성되어 있으며,
원본 5개 클래스를 다음과 같이 이진 분류 문제로 변환하였습니다.

| Original Label | Final Label |
| -------------- | ----------- |
| 0              | Normal      |
| 1~4            | Abnormal    |

---

## 강화학습 문제 정의

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

| Situation        | Reward |
| ---------------- | ------ |
| Continue         | -0.03  |
| Wrong Prediction | -6     |
| Correct Normal   | +1     |
| Correct Abnormal | +6     |

Continue 행동에는 관찰 비용을 부여하였으며,

이상 탐지 실패에는 더 큰 패널티를 적용하여

> **정확도와 조기성 간의 균형을 학습하도록 설계하였습니다.**

---

## 모델 구조

본 연구에서는 ECG 특징 추출 자체보다

> **판단 시점 학습(Decision Timing Learning)**

에 초점을 두었습니다.

따라서 복잡한 네트워크 대신 단순한 Q-Network를 사용하였습니다.

```
Input State
(ECG + Observed Ratio)

↓

FC(128)
ReLU

↓

FC(128)
ReLU

↓

Output (3 Actions)
```

---

## 실험 환경

* Framework : PyTorch
* Environment : Google Colab
* GPU : Tesla T4
* Episode : 3,000
* Batch Size : 64
* Learning Rate : 0.001
* Discount Factor : 0.99
* Seed : 42

---

## 주요 결과

### Reward 설계 비교

| Experiment |  Accuracy | Abnormal F1 | Avg. Observed Ratio |
| ---------- | --------: | ----------: | ------------------: |
| Exp1       |     0.811 |       0.340 |               0.700 |
| Exp2       |     0.835 |       0.436 |               0.873 |
| Exp3       | **0.897** |   **0.694** |           **0.748** |

최종 Reward 설계에서는

* Accuracy : 89.7%
* Abnormal F1 : 0.694
* Avg. Observed Ratio : 0.748

을 달성하였습니다.

이는

> **평균적으로 전체 ECG의 약 75%만 관찰하고도 이상 여부를 판단할 수 있었음을 의미합니다.**

즉,

ECG의 약 25%를 추가로 읽지 않고도 일정 수준 이상의 이상 탐지가 가능함을 확인하였습니다.

---

## 기존 모델과의 비교

| Model               | Abnormal F1 | Observed Ratio |
| ------------------- | ----------: | -------------: |
| Logistic Regression |        0.63 |           1.00 |
| LSTM                |        0.11 |           1.00 |
| MLP                 |        0.90 |           1.00 |
| DQN                 |        0.69 |           0.75 |

MLP는 가장 높은 분류 성능을 보였지만,

모든 샘플에 대해 전체 ECG를 관찰해야 했습니다.

반면 DQN은 일부 성능 손실을 감수하는 대신,

샘플마다 서로 다른 시점에 판단을 종료하는

> **동적 의사결정(Early Decision Making)**

이 가능하다는 점을 확인하였습니다.

---

## 한계 및 향후 연구

본 연구는 ECG 데이터를 활용하여 판단 시점 최적화 가능성을 검증하였으나, 몇 가지 한계가 존재합니다.

첫째,

ECG 데이터는 길이가 비교적 짧고 초기 이상 패턴이 뚜렷하게 나타나는 특성이 있어,

판단 시점 최적화의 효과를 충분히 검증하기에는 제한이 존재합니다.

둘째,

Reward 설계와 초기 탐색 과정에 따라 성능 편차가 발생하였으며,

보다 안정적인 정책 학습을 위해 Double DQN, Dueling DQN 등의 확장이 필요합니다.

셋째,

본 연구에서 제안한 접근은 의료 분야를 넘어

네트워크 이상 탐지, 장애 대응, 산업 설비 모니터링 등

> **무엇을 판단할 것인가"보다 "언제 판단할 것인가가 중요한 문제**

로 확장될 수 있습니다.

---

## 결론

본 연구는 ECG 조기 이상 탐지를 순차적 의사결정 문제로 재정의하고,

강화학습을 통해 판단 시점을 학습할 수 있는 가능성을 확인하였습니다.

최고 수준의 분류 성능을 달성한 연구는 아니지만,

> **제한된 정보만으로도 언제 판단할 것인지를 학습할 수 있다는 가능성을 제시했다는 점에서 의미가 있습니다.**

이는 향후 AI Agent 기반의 동적 의사결정 환경에서 활용될 수 있는 초기 연구로서의 가치를 가집니다.
