import re

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from intents import FALLBACK_ANSWER, INTENTS

# 가장 비슷한 예시 문장의 유사도가 이 값보다 낮으면 "이해 못함"으로 처리
THRESHOLD = 0.3

# 어느 카테고리에나 나오는 말. 비교 전에 지워서 핵심 단어로만 비교하게 한다.
FILLER_WORDS = [
    "어떻게", "언제", "오늘", "내일", "얼마", "어디", "혹시", "그냥", "너무", "아직",
    "하고 싶어요", "싶어요", "싶은데", "해주세요", "주세요", "할게요", "하려면",
    "하나요", "되나요", "있나요", "가능한가요", "가능해요", "인데요", "는데요",
    "이에요", "예요", "어때", "해요", "돼요", "나요", "요",
]
_FILLER_RE = re.compile("|".join(sorted(map(re.escape, FILLER_WORDS), key=len, reverse=True)))


def normalize(text):
    text = re.sub(r"[^\w\s]", " ", text)  # 물음표, 느낌표 등 제거
    return _FILLER_RE.sub(" ", text)


@st.cache_resource
def build_model():
    """예시 문장들을 글자 단위 TF-IDF 벡터로 바꿔 둔다 (앱 시작 시 한 번만 실행)."""
    sentences, labels = [], []
    for intent, data in INTENTS.items():
        for example in data["examples"]:
            sentences.append(example)
            labels.append(intent)

    # 글자 2~3개 단위로 비교하면 한국어 어미 변화에도 잘 맞는다
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), preprocessor=normalize)
    matrix = vectorizer.fit_transform(sentences)
    return vectorizer, matrix, labels


def classify(question):
    """질문과 가장 비슷한 예시 문장을 찾아 (카테고리, 유사도)를 돌려준다."""
    vectorizer, matrix, labels = build_model()
    scores = cosine_similarity(vectorizer.transform([question]), matrix)[0]
    best = scores.argmax()
    if scores[best] < THRESHOLD:
        return None, scores[best]
    return labels[best], scores[best]


def answer(question):
    intent, _ = classify(question)
    if intent is None:
        return FALLBACK_ANSWER
    return INTENTS[intent]["answer"]


st.title("🛒 쇼핑몰 고객센터")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 다시 그리기
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

question = st.chat_input("궁금한 점을 입력하세요")
if question:
    reply = answer(question)
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.chat_message("user").write(question)
    st.chat_message("assistant").write(reply)
