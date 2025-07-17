# 🤖 machine-learning-analyzer

비행기 실험 데이터를 기반으로 머신러닝을 활용한 예측/분석 웹앱입니다.

---

## 🧩 주요 기능

- 📥 종이컵 비행기, 고리 비행기, 직접 업로드용 엑셀 양식 제공
- 📂 실험 데이터 업로드 후 머신러닝 분석 가능 (여러 파일도 지원)
- 🧠 선형회귀 및 랜덤포레스트 알고리즘 지원
- 🛠 튜닝 옵션(n_estimators, max_depth) 및 교차검증(K-Fold) 설정
- 📊 성능 지표: R², RMSE, MAE, 교차검증 평균 R²
- 📈 예측 vs 실제 시각화 / 📉 변수별 관계 시각화 / 📌 변수 중요도 분석
- 🧪 새 입력 조건 예측 기능
- 🚨 데이터 오류 자동 검토 및 사용자 메시지 안내
- 📘 머신러닝 및 랜덤포레스트 개념 사이드바 설명 포함
- 🔗 공공 데이터 포털 추천 링크 제공

---

## 🚀 실행 방법

```bash
git clone https://github.com/yourname/machine-learning-analyzer.git
cd machine-learning-analyzer
pip install -r requirements.txt
streamlit run app.py
