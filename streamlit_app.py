# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import matplotlib
import io
import os
import re

from openpyxl import Workbook
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 한글 폰트 설정
font_path = "./NanumGothic.ttf"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams['font.family'] = font_name
    matplotlib.rcParams['font.family'] = font_name
else:
    matplotlib.rcParams['font.family'] = ['Malgun Gothic', 'AppleGothic', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# 잘못된 문자 제거 함수
def remove_illegal_characters(s):
    if isinstance(s, str):
        return re.sub(r'[\x00-\x1F]', '', s)
    return s

# 데이터 오류 검사 함수
def check_data_issues(df):
    messages = []
    nulls = df.isnull().sum()
    for col, cnt in nulls.items():
        if cnt > 0:
            messages.append(f"📌 `{col}` 컬럼에 결측치 {cnt}개가 있어요.")
    if '비행성능' in df.columns:
        outliers = df[(df['비행성능'] < 0) | (df['비행성능'] > 20)]
        if not outliers.empty:
            messages.append(f"🚨 비행성능 값이 0~20 범위를 벗어난 데이터가 {len(outliers)}개 있어요.")
    return messages

# 엑셀 템플릿 생성 함수
def generate_excel_with_two_sheets(experiment):
    wb = Workbook()
    ws_analysis = wb.active
    ws_analysis.title = remove_illegal_characters("분석용 데이터")
    ws_input = wb.create_sheet(remove_illegal_characters("원본 데이터"))

    if experiment == "종이컵 비행기":
        input_cols = ["번호", "모둠명", "안쪽 지름(cm)", "바깥쪽 지름(cm)", "반너비(cm)", "고무줄 감은 횟수",
                      "고무줄 늘어난 길이(cm)", "무게(g)", "날리는 높이(cm)",
                      "비행성능1", "비행성능2", "비행성능3", "비행성능4", "비행성능5"]
        analysis_cols = input_cols[2:9] + ["비행성능"]
        ws_analysis.append(analysis_cols)
        for i in range(2, 102):
            row = [f"='원본 데이터'!{chr(65 + input_cols.index(col))}{i}" if col != "비행성능" 
                   else f"=AVERAGE('원본 데이터'!J{i}:N{i})" for col in analysis_cols]
            ws_analysis.append(row)
        ws_input.append(input_cols)

    elif experiment == "고리 비행기":
        input_cols = ["번호", "모둠명", "앞 쪽 고리 지름(cm)", "앞 쪽 고리 두께(cm)", "뒤 쪽 고리 지름(cm)",
                      "뒤 쪽 고리 두께(cm)", "질량(g)", "고무줄길이(cm)", "무게 중심(cm)", "고무줄늘어난길이(cm)",
                      "비행성능1", "비행성능2", "비행성능3", "비행성능4", "비행성능5"]
        analysis_cols = input_cols[2:6] + ["질량(g)", "고무줄늘어난길이(cm)", "비행성능"]
        ws_analysis.append(analysis_cols)
        for i in range(2, 102):
            row = [f"='원본 데이터'!{chr(65 + input_cols.index(col))}{i}" if col != "비행성능" 
                   else f"=AVERAGE('원본 데이터'!K{i}:O{i})" for col in analysis_cols]
            ws_analysis.append(row)
        ws_input.append(input_cols)

    elif experiment == "직접 업로드":
        df_custom = pd.DataFrame(columns=["특성1", "특성2", "특성3", "예측하고 싶은 값"])
        stream = io.BytesIO()
        df_custom.to_excel(stream, index=False, sheet_name="분석용 데이터")
        stream.seek(0)
        return stream

    stream = io.BytesIO()
    wb.save(stream)
    stream.seek(0)
    return stream

# 제목
st.title("🤖 머신러닝 분석기")
st.markdown("""
> 여러 개의 엑셀 데이터를 올릴 수 있어요! 날짜나 파일명을 기준으로 자동으로 병합되며,
> 분석 결과는 하나로 통합해서 보여줘요 😊
""")

# 실험 유형 선택
experiment = st.selectbox("🔬 실험 종류를 선택하세요", ["종이컵 비행기", "고리 비행기", "직접 업로드"])

# 양식 다운로드
file_name = f"{experiment}_샘플_양식.xlsx"
towrite = generate_excel_with_two_sheets(experiment)
st.download_button("📥 샘플 엑셀 양식 다운로드", data=towrite, file_name=file_name)

# 엑셀 업로드
uploaded_files = st.file_uploader("📂 실험 엑셀 업로드 (여러 파일 가능, 분석용 데이터 시트 포함)", type=["xlsx"], accept_multiple_files=True)", type=["xlsx"])

if not uploaded_files:
    st.stop()

try:
    df_list = []
    for f in uploaded_files:
        temp_df = pd.read_excel(f, sheet_name="분석용 데이터")
        temp_df.columns = temp_df.columns.str.replace("
", " ").str.strip()
        temp_df = temp_df.select_dtypes(include=['number'])
        temp_df['파일명'] = f.name  # 파일 구분용
        df_list.append(temp_df)
    df = pd.concat(df_list, ignore_index=True)
    issues = check_data_issues(df)
    if issues:
        st.warning("❗ 데이터 확인 필요:")
        for msg in issues:
            st.markdown(f"- {msg}")
except Exception:
    st.error("❌ '분석용 데이터' 시트를 불러오는 데 실패했습니다.")
    st.stop()

# 분석 시작
st.subheader("📊 분석 결과")
columns = df.columns.tolist()
target_candidates = [c for c in columns if '성능' in c or '평균' in c or c.lower() in ['target', 'y']]
default_target = target_candidates[0] if target_candidates else columns[-1]

target_col = st.selectbox("🎯 예측할 종속변수", columns, index=columns.index(default_target))
feature_cols = st.multiselect("🧪 독립변수(입력값)", [c for c in columns if c != target_col], default=[c for c in columns if c != target_col])

st.sidebar.subheader("🧠 모델 설정")

st.sidebar.markdown("""
### 🤖 머신러닝이란?
머신러닝은 **컴퓨터가 데이터에서 스스로 패턴을 찾고**, 그걸 바탕으로 **예측이나 판단을 하게 만드는 기술**이에요. 

예를 들어,
- 종이컵 비행기가 잘 날아간 이유가 **무게** 때문인지, **고리의 크기** 때문인지 찾아주는 거예요.
- 새로운 조건에서도 얼마나 날 수 있을지 **예측**도 할 수 있어요!

학생 여러분은 실험 데이터를 통해 **'이런 조건일 때 더 잘 날더라!'** 하는 **과학적 근거**를 찾을 수 있어요 ✨
""")
model_option = st.sidebar.selectbox("머신러닝 알고리즘 선택", ["선형회귀", "랜덤포레스트"])
tuning = st.sidebar.checkbox("튜닝 사용", value=(model_option == "랜덤포레스트"))
kfolds = st.sidebar.slider("K-Fold 수 (교차검증)", 2, 10, 5)

if model_option == "랜덤포레스트" and tuning:
    n_estimators = st.sidebar.slider("n_estimators", 10, 300, 100, 10)
    max_depth = st.sidebar.slider("max_depth", 1, 30, 5)
else:
    n_estimators = 100
    max_depth = None

st.sidebar.markdown("""
### 📘 K-Fold 교차검증이란?
- 데이터를 여러 조각으로 나누어
- 여러 번 학습+시험을 반복하여
- 모델이 운 좋게 맞춘 게 아닌지 확인하는 방법입니다.
""")

st.sidebar.markdown("""
### 🌲 랜덤포레스트 설명
- `n_estimators`: 나무 개수 (많을수록 안정적)
- `max_depth`: 나무 깊이 (깊을수록 복잡, 너무 깊으면 과적합 위험)
""")

X = df[feature_cols]
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression() if model_option == "선형회귀" else RandomForestRegressor(
    n_estimators=n_estimators, max_depth=max_depth, random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
mae = mean_absolute_error(y_test, y_pred)
cv_score = cross_val_score(model, X, y, cv=kfolds, scoring='r2').mean()

st.success(f"✅ 테스트 R²: {r2:.2f} | RMSE: {rmse:.2f} | MAE: {mae:.2f} | 교차검증 R² 평균: {cv_score:.2f}")

# 예측 vs 실제
st.subheader("📈 예측 vs 실제")
fig1, ax1 = plt.subplots()
sns.regplot(x=model.predict(X), y=y, ax=ax1, ci=95, line_kws={"color": "blue"})
ax1.set_xlabel("예측값")
ax1.set_ylabel("실제값")
st.pyplot(fig1)

# 변수별 성능 관계
st.subheader("📉 독립변수별 성능 관계")
selected_feature = st.selectbox("🔍 분석할 변수 선택", feature_cols)
fig2, ax2 = plt.subplots()
sns.scatterplot(x=selected_feature, y=target_col, data=df, ax=ax2)
sns.regplot(x=selected_feature, y=target_col, data=df, ax=ax2, scatter=False, line_kws={"color": "red"})
st.pyplot(fig2)

# 변수 중요도
st.subheader("📌 변수 중요도")
if model_option == "랜덤포레스트":
    importance_df = pd.DataFrame({"변수": X.columns, "중요도": model.feature_importances_})
else:
    importance_df = pd.DataFrame({"변수": X.columns, "중요도": np.abs(model.coef_)})
importance_df = importance_df.sort_values(by="중요도", ascending=False)
fig3, ax3 = plt.subplots()
sns.barplot(data=importance_df, x="중요도", y="변수", ax=ax3)
st.pyplot(fig3)

# 새 입력값 예측
st.subheader("🧪 새 조건 입력 → 예측값")
input_data = {col: st.number_input(f"{col}", value=float(X[col].mean())) for col in feature_cols}
input_df = pd.DataFrame([input_data])
prediction = model.predict(input_df)[0]
st.success(f"📊 예측 결과: {prediction:.2f}")
