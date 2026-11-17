import streamlit as st


def answer(question):
    if "배송" in question:
        return "배송은 보통 2~3일 걸립니다."
    elif "환불" in question:
        return "환불은 마이페이지에서 신청할 수 있습니다."
    return "죄송합니다. 상담원에게 연결해 드릴게요."


st.title("🛒 쇼핑몰 고객센터")

question = st.chat_input("궁금한 점을 입력하세요")
if question:
    st.chat_message("user").write(question)
    st.chat_message("assistant").write(answer(question))