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
import openai

from openpyxl import Workbook
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 🔐 OpenAI API 키 설정
openai.api_key = "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"

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

# 🧪 메인 소개 페이지
st.set_page_config(layout="wide")
st.title("✈️ 비행기 실험 데이터 분석기")
col1, col2 = st.columns([1, 4])

with col1:
    st.markdown("""
    ## 📘 머신러닝이란?
    머신러닝은 컴퓨터가 **데이터를 통해 스스로 학습**하고 **예측을 수행**하는 기술이에요.

    - 예) 고리 크기, 무게, 회전수 등을 통해 비행 성능을 예측해요.
    - 우리의 실험 데이터도 머신러닝으로 분석할 수 있어요!

    ➡️ 아래에서 데이터를 업로드하고, 예측 모델을 설정해보세요!
    """)

with col2:
    st.markdown("""
    **머신러닝을 활용한 실험 데이터 분석 앱입니다.**

    - 왼쪽 메뉴에서 머신러닝 설명과 알고리즘을 이해할 수 있어요.
    - 아래에서 실험 데이터를 업로드하고 분석을 시작하세요 ✨
    """)

# AI 챗봇 응답 생성 함수
def get_chat_response(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"⚠️ 에러 발생: {e}"

# AI 도우미 영역
with st.expander("🤖 AI에게 질문하기 (머신러닝 관련 도우미)"):
    user_prompt = st.text_input("무엇이 궁금한가요? 예: '랜덤포레스트가 뭔가요?'", key="ai_chat")
    if user_prompt:
        with st.spinner("GPT가 답변 중입니다..."):
            answer = get_chat_response(user_prompt)
            st.markdown(f"📎 답변: {answer}")

# 실험 종류 선택 및 업로드
experiment = st.selectbox("🔬 실험 종류를 선택하세요", ["종이컵 비행기", "고리 비행기", "직접 업로드"])
file_name = f"{experiment}_샘플_양식.xlsx"
st.download_button("📥 샘플 엑셀 양식 다운로드", data=b"", file_name=file_name)  # 예시용 빈 바이트
uploaded_files = st.file_uploader("📂 실험 엑셀 업로드 (분석용 데이터 시트 포함)", type=["xlsx"], accept_multiple_files=True)

if uploaded_files:
    df_list = []
    for f in uploaded_files:
        try:
            df = pd.read_excel(f, sheet_name="분석용 데이터")
            df['파일명'] = f.name
            df_list.append(df)
        except ValueError:
            st.error(f"❌ '{f.name}' 파일에서 '분석용 데이터' 시트를 찾을 수 없습니다.")
    merged_df = pd.concat(df_list, ignore_index=True)
    st.success("✅ 파일 업로드 및 병합 완료")

    # 분석 대상 선택
    columns = merged_df.columns.tolist()
    target_col = st.selectbox("🎯 예측할 종속변수", columns)
    feature_cols = st.multiselect("🧪 독립변수(입력값)", [c for c in columns if c != target_col])

    # 모델 선택 및 튜닝
    model_option = st.selectbox("모델 선택", ["선형회귀", "랜덤포레스트"])
    if model_option == "랜덤포레스트":
        n_estimators = st.slider("n_estimators", 10, 300, 100)
        max_depth = st.slider("max_depth", 1, 30, 5)
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    else:
        model = LinearRegression()

    # 모델 학습 및 결과
    X = merged_df[feature_cols].select_dtypes(include=[np.number]).dropna()
    y = merged_df.loc[X.index, target_col].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write(f"✅ R²: {r2_score(y_test, y_pred):.2f} / RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f} / MAE: {mean_absolute_error(y_test, y_pred):.2f}")

    # 예측 vs 실제
    fig, ax = plt.subplots()
    sns.regplot(x=y_pred, y=y_test, ax=ax)
    ax.set_xlabel("예측값")
    ax.set_ylabel("실제값")
    st.pyplot(fig)
