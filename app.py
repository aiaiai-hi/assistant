import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import openai

# ── Загрузка индекса ────────────────────────────────────────────
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    return FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )

# ── Запрос к Qwen ───────────────────────────────────────────────
def ask_qwen(prompt: str, api_key: str) -> str:
    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://api.vsellm.ru/v1"
    )
    response = client.chat.completions.create(
        model="qwen/qwen3-vl-8b-instruct",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ── RAG ─────────────────────────────────────────────────────────
def rag_answer(question: str, vectorstore, api_key: str, k: int = 3):
    docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n---\n\n".join([d.page_content for d in docs])

    prompt = f"""Ты помощник по работе с системой Бизнес-Глоссарий.
Отвечай только на основе контекста ниже.
Если ответа в контексте нет — скажи об этом прямо.

Контекст:
{context}

Вопрос: {question}
Ответ:"""

    return ask_qwen(prompt, api_key), docs

# ── UI ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Помощник БГ", page_icon="🤖")
st.title("🤖 Помощник по Бизнес-Глоссарию")

api_key = st.secrets["QWEN_API_KEY"]

vectorstore = load_vectorstore()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Задайте вопрос..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Ищу ответ..."):
            try:
                answer, source_docs = rag_answer(question, vectorstore, api_key)
            except Exception as e:
                answer = f"Ошибка: {str(e)}"
                source_docs = []

        st.markdown(answer)

        if source_docs:
            with st.expander("📄 Источники"):
                for i, doc in enumerate(source_docs, 1):
                    st.markdown(f"**{i}. {doc.metadata.get('topic', '')}**")
                    st.markdown(f"{doc.page_content[:200]}...")
                    st.markdown(f"*Файл: {doc.metadata.get('source_file', '')}*")

    st.session_state.messages.append({"role": "assistant", "content": answer})
