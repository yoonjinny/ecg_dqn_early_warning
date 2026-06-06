# 프로젝트 기획서 초안

## 제목
DQN 기반 ECG 조기 이상 경보 Agent 설계 및 구현

## 핵심 아이디어
기존 ECG 분류는 전체 ECG를 본 후 정상/이상을 예측한다. 본 프로젝트는 ECG를 시간 순서대로 일부만 관찰하면서, agent가 더 관찰할지 또는 정상/이상으로 즉시 판단할지를 결정하는 강화학습 문제로 재정의한다.

## 문제 정의
- Environment: ECG heartbeat sample
- Agent: DQN early-warning agent
- State: 현재까지 관찰한 ECG segment + zero padding + 관찰 비율
- Action: Continue / Predict Normal / Predict Abnormal
- Reward: 빠르고 정확한 판단에 높은 보상, 오분류에는 큰 패널티

## 기대 결과
DQN은 일반 분류기보다 F1-score가 다소 낮을 수 있지만, 전체 ECG의 100%를 보지 않고 평균 30~60% 정도만 관찰하고도 이상 경보를 수행할 수 있음을 보이는 것이 목표다.

## 실험 비교
- Baseline: Logistic Regression 또는 MLP, 전체 ECG 100% 관찰
- Proposed: DQN, 일부 ECG 관찰 후 조기 판단

## 발표 포인트
이 프로젝트의 강점은 분류 문제를 단순히 지도학습으로 푸는 것이 아니라, 조기 판단 시점까지 action으로 포함하여 강화학습 문제로 설계했다는 점이다.
