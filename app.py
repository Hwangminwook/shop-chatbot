import importlib
import io
import re
from datetime import datetime

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

import intents
import test_data

# 배포 서버가 예전 파일을 메모리에 들고 있지 않도록, 실행할 때마다 다시 읽는다.
importlib.reload(intents)
importlib.reload(test_data)
INTENTS = intents.INTENTS
FALLBACK_ANSWER = intents.FALLBACK_ANSWER
TEST_DATA = test_data.TEST_DATA
CATEGORIES = list(INTENTS)

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


# ---------------------------------------------------------------- 모델

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
def build_model(sentences, labels):
    """예시 문장으로 TF-IDF 벡터와 카테고리 분류 모델을 만든다.

    예시 문장이 같으면 저장해 둔 결과를 다시 쓰고, 바뀌면 새로 만든다.
    """
    # 글자 2~3개 단위로 비교하면 한국어 어미 변화에도 잘 맞는다
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), preprocessor=normalize)
    matrix = vectorizer.fit_transform(sentences)

    # 카테고리별 확률을 내기 위한 분류 모델
    classifier = LogisticRegression(C=10, max_iter=1000)
    classifier.fit(matrix, [category for category, _ in labels])
    return vectorizer, matrix, classifier


def predict(question):
    """질문을 분류한다.

    1. 분류 모델로 카테고리별 확률을 구해 가장 높은 카테고리를 고른다.
    2. 그 카테고리의 예시 문장 중 가장 비슷한 문장의 세부 질문을 고른다.
    3. 어떤 예시 문장과도 충분히 비슷하지 않으면 카테고리/세부 질문을 None 으로 둔다.
    """
    sentences, labels = collect_examples()
    vectorizer, matrix, classifier = build_model(sentences, labels)

    vector = vectorizer.transform([question])
    probs = dict(zip(classifier.classes_, classifier.predict_proba(vector)[0]))
    probs = {category: float(probs.get(category, 0.0)) for category in CATEGORIES}
    similarities = cosine_similarity(vector, matrix)[0]

    result = {"probs": probs, "category": None, "topic": None, "similarity": float(similarities.max())}
    if similarities.max() < THRESHOLD:
        return result

    category = max(probs, key=probs.get)
    in_category = [i for i, (c, _) in enumerate(labels) if c == category]
    best = max(in_category, key=lambda i: similarities[i])
    result.update(category=category, topic=labels[best][1], similarity=float(similarities[best]))
    return result


def answer_for(result):
    if result["category"] is None:
        return FALLBACK_ANSWER
    return INTENTS[result["category"]][result["topic"]]["answer"]


# ---------------------------------------------------------------- 평가 / 엑셀

def evaluate():
    """테스트 문장 전체를 분류해서 결과 표를 만든다."""
    rows = []
    for question, want_category, want_topic in TEST_DATA:
        result = predict(question)
        row = {
            "질문": question,
            "정답 카테고리": want_category or "(답변 안 함)",
            "예측 카테고리": result["category"] or "(답변 안 함)",
            "정답 세부 질문": want_topic or "(답변 안 함)",
            "예측 세부 질문": result["topic"] or "(답변 안 함)",
        }
        row["카테고리 정답"] = row["정답 카테고리"] == row["예측 카테고리"]
        row["세부 질문 정답"] = row["카테고리 정답"] and row["정답 세부 질문"] == row["예측 세부 질문"]
        row["유사도"] = round(result["similarity"], 3)
        for category, p in result["probs"].items():
            row[f"{category} 확률"] = round(p, 3)
        rows.append(row)
    return pd.DataFrame(rows)


def to_excel(df, sheet_name):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return buffer.getvalue()


XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def show_probs(probs):
    for category, p in sorted(probs.items(), key=lambda kv: kv[1], reverse=True):
        st.progress(p, text=f"{category} {p:.1%}")


# ---------------------------------------------------------------- 화면

st.title("🛒 쇼핑몰 고객센터")

if "messages" not in st.session_state:
    st.session_state.messages = []  # 화면에 다시 그릴 대화
if "records" not in st.session_state:
    st.session_state.records = []  # 엑셀로 저장할 분류 기록

chat_tab, test_tab = st.tabs(["💬 상담", "📊 정확도 테스트"])

with chat_tab:
    question = st.chat_input("궁금한 점을 입력하세요")
    if question:
        result = predict(question)
        reply = answer_for(result)
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.messages.append({"role": "assistant", "content": reply, "probs": result["probs"]})
        st.session_state.records.append({
            "시각": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "질문": question,
            "예측 카테고리": result["category"] or "(답변 안 함)",
            "예측 세부 질문": result["topic"] or "(답변 안 함)",
            "유사도": round(result["similarity"], 3),
            **{f"{c} 확률": round(p, 3) for c, p in result["probs"].items()},
            "답변": reply,
        })

    last = len(st.session_state.messages) - 1
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "probs" in message:
                with st.expander("카테고리별 예측확률", expanded=(i == last)):
                    show_probs(message["probs"])

    if st.session_state.records:
        st.download_button(
            "📥 상담 기록 엑셀로 저장",
            data=to_excel(pd.DataFrame(st.session_state.records), "상담 기록"),
            file_name=f"상담기록_{datetime.now():%Y%m%d_%H%M}.xlsx",
            mime=XLSX_MIME,
        )

with test_tab:
    st.caption(
        f"학습에 쓰지 않은 테스트 문장 {len(TEST_DATA)}개로 측정한 결과예요. "
        "테스트 문장은 test_data.py 에서 추가·수정할 수 있어요."
    )
    df = evaluate()
    col1, col2, col3 = st.columns(3)
    col1.metric("카테고리 정확도", f"{df['카테고리 정답'].mean():.1%}")
    col2.metric("세부 질문 정확도", f"{df['세부 질문 정답'].mean():.1%}")
    col3.metric("틀린 문장", f"{(~df['세부 질문 정답']).sum()}개")

    only_wrong = st.toggle("틀린 문장만 보기")
    st.dataframe(df[~df["세부 질문 정답"]] if only_wrong else df, hide_index=True)

    st.download_button(
        "📥 분류 결과 엑셀로 저장",
        data=to_excel(df, "분류 결과"),
        file_name=f"분류결과_{datetime.now():%Y%m%d_%H%M}.xlsx",
        mime=XLSX_MIME,
    )
