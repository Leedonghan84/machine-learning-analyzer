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
st.title("✈️ 비행기 실험 데이터 분석기")
st.markdown("""
**머신러닝을 활용한 실험 데이터 분석 앱입니다.**

- 왼쪽 메뉴에서 알고리즘과 튜닝 조건을 조절할 수 있어요.
- 아래에서 실험 데이터를 업로드하고 머신러닝 분석을 시작하세요 ✨
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

# 🔽 이하 기존 코드 유지 (업로드, 분석 등 계속 연결)...
