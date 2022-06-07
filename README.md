# PUBG 최종 순위 예측

- https://www.kaggle.com/competitions/pubg-finish-placement-prediction

## 폴더 구조

```
├─ data: 데이터
│  ├─ raw: 원본 데이터
│  │  ├─ train_V2.csv
│  │  └─ test_V2.csv
│  ├─ interim: 전처리 중간 과정 데이터
│  └─ processed: 전처리가 끝나 모델링에 사용 가능한 데이터
├─ model: 학습된 모델 저장 (Serialized)
├─ experiments: 실험 설정 파일 (모델, 하이퍼파라미터, 저장경로)
├─ src: 소스 코드
│  ├─ features: 전처리 코드
│  └─ model: 모델 코드
└─ notebooks: 데이터 탐색, 소통용 ipynb
```