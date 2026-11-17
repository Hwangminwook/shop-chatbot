import importlib
import re

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import intents

# 배포 서버가 예전 intents.py를 메모리에 들고 있지 않도록, 실행할 때마다 다시 읽는다.
importlib.reload(intents)
INTENTS = intents.INTENTS
FALLBACK_ANSWER = intents.FALLBACK_ANSWER

# 가장 비슷한 예시 문장의 유사도가 이 값보다 낮으면 "이해 못함"으로 처리
THRESHOLD = 0.3

# 어느 카테고리에나 나오는 말. 비교 전에 지워서 핵심 단어로만 비교하게 한다.
FILLER_WORDS = [
    "어떻게", "오늘", "어디", "혹시", "그냥", "너무", "뭐",
    "하고 싶어요", "싶어요", "싶은데", "해주세요", "주세요", "할게요", "하려면",
    "하나요", "되나요", "있나요", "가능한가요", "가능해요", "인데요", "는데요",
    "이에요", "예요", "어때", "해요", "돼요", "나요", "요",
]
_FILLER_RE = re.compile("|".join(sorted(map(re.escape, FILLER_WORDS), key=len, reverse=True)))


def normalize(text):
    text = re.sub(r"[^\w\s]", " ", text)  # 물음표, 느낌표 등 제거
    return _FILLER_RE.sub(" ", text)


def collect_examples():
    """INTENTS에서 (예시 문장들, 각 문장의 (카테고리, 세부 질문))을 뽑는다."""
    sentences, labels = [], []
    for category, topics in INTENTS.items():
        for topic, data in topics.items():
            for example in data["examples"]:
                sentences.append(example)
                labels.append((category, topic))
    return tuple(sentences), tuple(labels)


@st.cache_resource
def build_model(sentences):
    """예시 문장들을 글자 단위 TF-IDF 벡터로 바꿔 둔다.

    예시 문장이 같으면 저장해 둔 결과를 다시 쓰고, 바뀌면 새로 만든다.
    """
    # 글자 2~3개 단위로 비교하면 한국어 어미 변화에도 잘 맞는다
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), preprocessor=normalize)
    matrix = vectorizer.fit_transform(sentences)
    return vectorizer, matrix


def classify(question):
    """질문과 가장 비슷한 예시 문장을 찾아 ((카테고리, 세부 질문), 유사도)를 돌려준다."""
    sentences, labels = collect_examples()
    vectorizer, matrix = build_model(sentences)
    scores = cosine_similarity(vectorizer.transform([question]), matrix)[0]
    best = scores.argmax()
    if scores[best] < THRESHOLD:
        return None, scores[best]
    return labels[best], scores[best]


def answer(question):
    intent, _ = classify(question)
    if intent is None:
        return FALLBACK_ANSWER
    category, topic = intent
    return INTENTS[category][topic]["answer"]


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
