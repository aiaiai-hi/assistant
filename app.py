"""
Глосси — ИИ-ассистент по Бизнес-Глоссарию
v10: уточнение типа отчёта/запроса, три уровня детализации, убрана подсказка шагового режима
"""

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import openai
from supabase import create_client, Client
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
# КОНФИГ
# ═══════════════════════════════════════════════════════════════
MODEL_NAME   = "qwen/qwen3-vl-30b-a3b-instruct"
EMBED_MODEL  = "intfloat/multilingual-e5-large"
FAISS_PATH   = "faiss_index"
API_BASE_URL = "https://api.vsellm.ru/v1"
TOP_K        = 10

# Картинка из репозитория
GLOSSY_IMG_URL = "https://raw.githubusercontent.com/aiaiai-hi/assistant/main/assets/glossy.png"
# ── Добавить в секцию КОНФИГ рядом с GLOSSY_IMG_URL ──────────
GLOSSY_VIDEO_URL = "https://raw.githubusercontent.com/aiaiai-hi/assistant/main/assets/glossy_intro.mp4"

# Быстрые запросы — отображаются кнопками
QUICK_QUESTIONS = [
    "Какие бывают виды запросов в БГ?",
    "Как зарегистрировать новый отчёт?",
    "Как автоматизировать отчёт?",
    "Что такое атрибутный состав?",
    "Как подобрать термин к атрибуту?",
    "Проведи меня по процессу шаг за шагом",
]

SYSTEM_PROMPT = """Ты — Глосси, ИИ-ассистент по работе с отчётами в Бизнес-Глоссарии (БГ) Банка.
 
## ГЛАВНОЕ ПРАВИЛО
Отвечай СТРОГО на основе фрагментов из базы знаний (раздел «Контекст» ниже).
- НЕ добавляй шаги, поля, кнопки или детали, которых НЕТ в контексте.
- НЕ перефразируй названия элементов интерфейса: если в контексте «нажать кнопку Создать» — пиши именно так.
- Если ответа в контексте нет — честно скажи об этом.
 
## Твоя личность
- Общаешься тепло, профессионально, без лишней воды.
- Если вопрос неоднозначный — уточняешь, не угадываешь.
 
## Квалификация запроса
В начале каждого ответа ОБЯЗАТЕЛЬНО определи тематику одной строкой:
`🏷️ Тема: [название темы]`
Темы: Регистрация отчёта | Актуализация | Исключение отчёта | Смена владельца | Автоматизация | Атрибутный состав | Навигация и поиск | Реестр отчётов | Общие вопросы | Другое
 
## УТОЧНЕНИЕ ТИПА ЗАПРОСА
 
Если пользователь спрашивает КАК СДЕЛАТЬ что-то в БГ — проверь, указан ли тип запроса в сообщении.
 
Слова-триггеры (НЕ переспрашивай):
- «новый отчёт», «зарегистрировать», «создать отчёт» → Регистрация нового отчёта
- «актуализировать», «обновить отчёт», «изменить отчёт» → Актуализация
- «автоматизировать», «BIQ», «бизнес-инициатива» → Автоматизация
- «исключить», «удалить отчёт» → Исключение
- «сменить владельца» → Смена владельца
 
Уточняй ТОЛЬКО если тип реально не ясен (например: «как создать запрос» — непонятно какой).
 
## РАБОТА С ПОШАГОВЫМИ ИНСТРУКЦИЯМИ
 
### КРИТИЧЕСКИ ВАЖНО: соблюдай порядок шагов
Фрагменты в контексте содержат поля `step_number` и `process_name`.
- Когда описываешь процесс — выводи шаги СТРОГО В ПОРЯДКЕ step_number (1, 2, 3…).
- НЕ пропускай шаги, даже если они кажутся очевидными.
- НЕ объединяй шаги, если они разделены в контексте.
- Если фрагмент с step_number=0 — это обзор процесса, используй его для общего ответа.
 
### Три уровня детализации
Когда пользователь просит объяснить процесс — спроси уровень:
 
«Как вам удобнее?
📋 **Кратко** — все этапы одним списком
📝 **По шагам** — краткое описание каждого шага
🔍 **Детально** — разбираем каждый шаг подробно, по одному»
 
**📋 Кратко:** Перечисли все этапы нумерованным списком. На каждый — 1 строка. Только из контекста.
 
**📝 По шагам:** Для каждого шага: название + 2–4 ключевых действия. Всё сразу. Используй ТОЛЬКО шаги из контекста.
 
**🔍 Детально:** Выдавай ОДИН шаг за раз. Воспроизводи содержимое из контекста ДОСЛОВНО. Нумеруй: «Шаг 1 из N». В конце добавь: `[NEXT_STEP_AVAILABLE]`. На «следующий»/«дальше»/«далее» — выдай следующий шаг.
 
## Что ты умеешь
1. Отвечать на вопросы по работе с отчётами в БГ
2. Объяснять типы запросов и типы отчётов
3. Проводить по процессу на выбранном уровне детализации
4. Объяснять заполнение полей, карточки запроса, атрибутного состава
5. Подсказать, где найти реестр отчётов (БГ или платформа УОЛ)
 
## Что ты НЕ умеешь
- Давать информацию по конкретным отчётам подразделений
- Генерировать атрибутный состав (→ раздел «Сформировать атрибуты» на платформе УОЛ)
- Создавать карточки запросов в системе
 
## Если ответа нет в контексте
Скажи: «У меня нет информации по этому вопросу в базе знаний. Обратитесь к команде Управления отчётным ландшафтом или Аналитики данных — контакты в разделе «Контакты».»
Не придумывай ответ.
"""


# ═══════════════════════════════════════════════════════════════
# ПОДКЛЮЧЕНИЯ
# ═══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.load_local(
        FAISS_PATH, embeddings, allow_dangerous_deserialization=True
    )

@st.cache_resource(show_spinner=False)
def get_supabase() -> Client:
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"],
    )


# ═══════════════════════════════════════════════════════════════
# LLM
# ═══════════════════════════════════════════════════════════════
def ask_qwen(messages: list, api_key: str) -> str:
    client = openai.OpenAI(
        api_key=api_key,
        base_url=API_BASE_URL,
        timeout=90,
    )
    # Добавляем /no_think в последнее сообщение пользователя
    patched = []
    for m in messages:
        if m["role"] == "user":
            patched.append({"role": "user", "content": "/no_think " + m["content"]})
        else:
            patched.append(m)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=patched,
        max_tokens=4096,
    )
    text = response.choices[0].message.content or ""
    # Подстраховка
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text


# ═══════════════════════════════════════════════════════════════
# RAG
# ═══════════════════════════════════════════════════════════════

# Маппинг триггеров → process_type для точного поиска
PROCESS_TRIGGERS = {
    "регистрац": "Регистрация нового отчёта",
    "новый отчёт": "Регистрация нового отчёта",
    "новый отчет": "Регистрация нового отчёта",
    "актуализ": "Актуализация отчёта",
    "исключ": "Исключение отчёта",
    "удал": "Исключение отчёта",
    "смен": "Смена владельца отчёта",
    "владел": "Смена владельца отчёта",
    "автоматиз": "Автоматизация отчёта",
    "атрибут": "Атрибутный состав",
    "АС ": "Атрибутный состав",
}

def detect_process_type(text: str) -> str | None:
    """Определяет process_type по тексту запроса."""
    text_lower = text.lower()
    for trigger, ptype in PROCESS_TRIGGERS.items():
        if trigger.lower() in text_lower:
            return ptype
    return None

def rag_answer(question: str, vectorstore, api_key: str, k: int = TOP_K):
    """RAG-ответ с сортировкой чанков по step_number и нумерацией фрагментов."""
 
    # Определяем тип процесса для расширения поиска
    process_type = detect_process_type(question)
 
    # Основной поиск по вопросу
    results = vectorstore.similarity_search_with_score(question, k=k)
 
    # Если определили тип процесса — добавляем целевой поиск
    if process_type:
        extra_queries = [
            f"шаги процесса {process_type} создание карточки запроса",
            f"{process_type} черновик редактировать местоположение сохранить отправить",
        ]
        seen = {r[0].page_content for r in results}
        for eq in extra_queries:
            extra = vectorstore.similarity_search_with_score(eq, k=k)
            for doc, score in extra:
                if doc.page_content not in seen:
                    results.append((doc, score))
                    seen.add(doc.page_content)
        # Берём топ-k по score (меньше = лучше в FAISS L2)
        results = sorted(results, key=lambda x: x[1])[:k]
 
    docs, raw = zip(*results) if results else ([], [])
    scores = [round(float(1 / (1 + d)), 3) for d in raw]
    avg_score = round(sum(scores) / len(scores), 3) if scores else 0.0
 
    # ─── КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: сортировка по step_number ───
    # Собираем (doc, score) и сортируем: сначала по process_name, потом по step_number
    doc_score_pairs = list(zip(docs, scores))
    doc_score_pairs.sort(key=lambda pair: (
        pair[0].metadata.get("process_name", ""),
        pair[0].metadata.get("step_number", 99),
    ))
    docs_sorted = [p[0] for p in doc_score_pairs]
    scores_sorted = [p[1] for p in doc_score_pairs]
 
    # ─── КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: нумерованный контекст с metadata ───
    context_parts = []
    for i, doc in enumerate(docs_sorted, 1):
        meta = doc.metadata
        header = f"[Фрагмент {i}"
        if meta.get("process_name"):
            header += f" — {meta['process_name']}"
        if meta.get("step_number") and meta["step_number"] > 0:
            header += f", шаг {meta['step_number']}"
        header += "]"
        context_parts.append(f"{header}\n{doc.page_content}")
 
    context = "\n\n---\n\n".join(context_parts)
    no_answer_marker = "ОТВЕТА_НЕТ"
 
    messages = [
        {
            "role": "system",
            "content": (
                SYSTEM_PROMPT
                + f"\n\n## Контекст из базы знаний\n{context}"
                + f"\n\nЕсли ответа в контексте нет — напиши ровно одно слово: {no_answer_marker}"
            ),
        }
    ]
    for msg in st.session_state.messages[-6:]:
        if msg["role"] in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})
 
    answer = ask_qwen(messages, api_key)
    no_answer = no_answer_marker.lower() in answer.lower()
 
    # Детектируем пошаговый режим
    next_step_available = "[NEXT_STEP_AVAILABLE]" in answer
    answer_clean = answer.replace("[NEXT_STEP_AVAILABLE]", "").strip()
 
    # Извлекаем тему
    topic = "Другое"
    for line in answer_clean.split("\n"):
        if "🏷️ Тема:" in line:
            topic = line.replace("🏷️ Тема:", "").strip()
            break
 
    return answer_clean, list(docs_sorted), scores_sorted, avg_score, no_answer, next_step_available, topic


# ═══════════════════════════════════════════════════════════════
# SUPABASE
# ═══════════════════════════════════════════════════════════════
def db_insert_log(question, answer, avg_score, no_answer, sources, topic) -> int | None:
    try:
        res = get_supabase().table("chat_logs").insert({
            "question" : question,
            "answer"   : answer,
            "avg_score": float(avg_score),
            "no_answer": no_answer,
            "feedback" : None,
            "sources"  : sources,
        }).execute()
        return res.data[0]["id"] if res.data else None
    except Exception as e:
        st.toast(f"⚠️ Не удалось сохранить в БД: {e}", icon="⚠️")
        return None

def db_update_feedback(row_id: int, feedback: str):
    try:
        get_supabase().table("chat_logs").update(
            {"feedback": feedback}
        ).eq("id", row_id).execute()
    except Exception as e:
        st.toast(f"⚠️ Не удалось обновить оценку: {e}", icon="⚠️")

@st.cache_data(ttl=30, show_spinner=False)
def db_load_logs():
    try:
        res = get_supabase().table("chat_logs") \
            .select("*").order("created_at").execute()
        return res.data or []
    except Exception:
        return []

@st.cache_data(ttl=30, show_spinner=False)
def db_load_metrics():
    logs  = db_load_logs()
    total = len(logs)
    if total == 0:
        return {"total": 0, "likes": 0, "dislikes": 0, "no_answer": 0, "avg_score": 0.0}
    likes    = sum(1 for r in logs if r["feedback"] == "like")
    dislikes = sum(1 for r in logs if r["feedback"] == "dislike")
    no_ans   = sum(1 for r in logs if r["no_answer"])
    avg_sc   = round(sum(float(r["avg_score"] or 0) for r in logs) / total, 3)
    return {"total": total, "likes": likes, "dislikes": dislikes,
            "no_answer": no_ans, "avg_score": avg_sc}


# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════
def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_metrics" not in st.session_state:
        st.session_state.session_metrics = {
            "total": 0, "likes": 0, "dislikes": 0,
            "no_answer": 0, "_score_sum": 0.0, "avg_score": 0.0,
        }
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "chat"
    if "scroll_to_last_question" not in st.session_state:
        st.session_state.scroll_to_last_question = False
    if "next_step_mode" not in st.session_state:
        st.session_state.next_step_mode = False   # пошаговый режим активен
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None  # вопрос от кнопки

def update_session_metrics(no_answer: bool, avg_score: float):
    m = st.session_state.session_metrics
    m["total"]      += 1
    m["_score_sum"] += avg_score
    m["avg_score"]   = round(m["_score_sum"] / m["total"], 3)
    if no_answer:
        m["no_answer"] += 1

def update_session_feedback(old_fb, new_fb):
    m = st.session_state.session_metrics
    if old_fb == "like":      m["likes"]    = max(0, m["likes"] - 1)
    elif old_fb == "dislike": m["dislikes"] = max(0, m["dislikes"] - 1)
    if new_fb == "like":      m["likes"]    += 1
    elif new_fb == "dislike": m["dislikes"] += 1


# ═══════════════════════════════════════════════════════════════
# СТИЛИ
# ═══════════════════════════════════════════════════════════════
def inject_styles():
    st.markdown("""
    <style>
    .stApp > header { display: none !important; }
    #root > div:first-child { padding-top: 0 !important; }
    .stApp { background: #f0fdf8; }
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 1200px;
    }

    /* хедер */
    .glossy-header {
        background: linear-gradient(135deg, #065f46 0%, #10b981 60%, #34d399 100%);
        border-radius: 16px; padding: 0 2rem 0 0;
        margin-top: 0.5rem; margin-bottom: 1rem; color: white;
        display: flex; align-items: flex-end; gap: 1.5rem;
        position: relative; overflow: hidden; min-height: 120px;
    }
    .glossy-header-text {
        padding: 1.25rem 0 1.25rem 0;
        display: flex; flex-direction: column; justify-content: flex-end;
    }
    .glossy-header-text h1   { margin: 0; font-size: 1.8rem; font-weight: 700; line-height: 1.1; }
    .glossy-header-text .subtitle { margin: 0.1rem 0 0; opacity: 0.9; font-size: 0.82rem; font-weight: 600; }
    .glossy-header-text .desc     { margin: 0.15rem 0 0; opacity: 0.75; font-size: 0.8rem; }
    .glossy-header-img {
        width: auto; height: 130px; object-fit: contain;
        flex-shrink: 0; align-self: flex-end;
    }

    /* метрики */
    .metric-row { display: flex; gap: 0.7rem; margin-bottom: 1rem; flex-wrap: wrap; }
    .metric-card {
        background: white; border: 1px solid #d1fae5; border-radius: 12px;
        padding: 0.6rem 1rem; flex: 1; min-width: 100px;
        text-align: center; box-shadow: 0 1px 4px rgba(16,185,129,.07);
    }
    .metric-card .val { font-size: 1.4rem; font-weight: 700; color: #065f46; line-height: 1.1; }
    .metric-card .lbl { font-size: 0.68rem; color: #6b7280; margin-top: 0.15rem; }

    /* кнопки быстрых запросов */
    .quick-btn-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.5rem 0 1rem; }

    /* score bar */
    .score-bar-wrap { background: #e5e7eb; border-radius: 4px; height: 5px; margin-top: 3px; }
    .score-bar      { background: #10b981; border-radius: 4px; height: 5px; }

    .no-answer-box {
        background: #fef9c3; border: 1px solid #fde68a; border-radius: 8px;
        padding: 0.4rem 0.8rem; font-size: 0.83rem; color: #92400e; margin-top: 0.35rem;
    }

    /* пошаговый режим */
    .step-box {
        background: #ecfdf5; border: 1px solid #10b981; border-radius: 10px;
        padding: 0.5rem 1rem; margin-top: 0.5rem; font-size: 0.83rem; color: #065f46;
    }

    .fb-like    { color: #10b981; font-weight: 600; }
    .fb-dislike { color: #ef4444; font-weight: 600; }
    .fb-none    { color: #d1d5db; }

    #last-question { scroll-margin-top: 80px; }
    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# СКРОЛЛ
# ═══════════════════════════════════════════════════════════════
def scroll_to_last_question():
    st.components.v1.html("""
    <script>
        setTimeout(function() {
            const el = window.parent.document.getElementById('last-question');
            if (el) { el.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
        }, 300);
    </script>
    """, height=0)


# ═══════════════════════════════════════════════════════════════
# НАВИГАЦИЯ
# ═══════════════════════════════════════════════════════════════
def render_nav():
    c1, c2, c3, _ = st.columns([1, 1, 1.4, 4])
    with c1:
        if st.button("💬 Чат",
                     type="primary" if st.session_state.active_tab == "chat" else "secondary",
                     use_container_width=True):
            st.session_state.active_tab = "chat"
            st.rerun()
    with c2:
        if st.button("ℹ️ О Глосси",
                     type="primary" if st.session_state.active_tab == "about" else "secondary",
                     use_container_width=True):
            st.session_state.active_tab = "about"
            st.rerun()
    with c3:
        if st.button("📊 Статистика",
                     type="primary" if st.session_state.active_tab == "stats" else "secondary",
                     use_container_width=True):
            st.session_state.active_tab = "stats"
            st.rerun()


# ═══════════════════════════════════════════════════════════════
# КОМПОНЕНТЫ
# ═══════════════════════════════════════════════════════════════
def render_assistant_message(content, log_id, avg_score=0.0,
                              no_answer=False, docs=None, scores=None,
                              next_step=False):
    st.markdown(content)

    if no_answer:
        st.markdown(
            '<div class="no-answer-box">⚠️ Ответ не найден в базе знаний. '
            'Попробуйте переформулировать вопрос.</div>',
            unsafe_allow_html=True,
        )

    # кнопки пошагового режима — без текстовой подсказки
    if next_step:
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("▶ Следующий шаг", key=f"next_{log_id}", use_container_width=True):
                st.session_state.pending_question = "следующий шаг"
                st.rerun()
        with col2:
            if st.button("↩ К началу", key=f"restart_{log_id}", use_container_width=True):
                st.session_state.pending_question = "начни сначала"
                st.rerun()

    if docs:
        with st.expander(f"📄 Источники  ·  avg score {avg_score:.2f}"):
            for i, (doc, score) in enumerate(zip(docs, scores or []), 1):
                topic = doc.metadata.get("topic", "—")
                src   = doc.metadata.get("source_file", "")
                bar_w = int(score * 100)
                st.markdown(f"**{i}. {topic}** — `{src}`")
                st.markdown(
                    f'<div class="score-bar-wrap">'
                    f'<div class="score-bar" style="width:{bar_w}%"></div></div>'
                    f'<small style="color:#6b7280">релевантность: {score}</small>',
                    unsafe_allow_html=True,
                )
                st.caption(doc.page_content[:200] + "…")

    if log_id is None:
        return

    cur_fb = next(
        (m["feedback"] for m in st.session_state.messages
         if m.get("log_id") == log_id),
        None,
    )
    c1, c2, _ = st.columns([2, 2, 6])
    with c1:
        lbl = "✅ Помогло" if cur_fb == "like" else "👍 Помогло"
        if st.button(lbl, key=f"like_{log_id}", use_container_width=True):
            db_update_feedback(log_id, "like")
            update_session_feedback(cur_fb, "like")
            for m in st.session_state.messages:
                if m.get("log_id") == log_id:
                    m["feedback"] = "like"
            db_load_logs.clear(); db_load_metrics.clear()
            st.rerun()
    with c2:
        lbl = "❌ Не помогло" if cur_fb == "dislike" else "👎 Не помогло"
        if st.button(lbl, key=f"dis_{log_id}", use_container_width=True):
            db_update_feedback(log_id, "dislike")
            update_session_feedback(cur_fb, "dislike")
            for m in st.session_state.messages:
                if m.get("log_id") == log_id:
                    m["feedback"] = "dislike"
            db_load_logs.clear(); db_load_metrics.clear()
            st.rerun()


# ═══════════════════════════════════════════════════════════════
# СТРАНИЦА — ЧАТ
# ═══════════════════════════════════════════════════════════════
def page_chat(vectorstore, api_key):
    # приветственное сообщение
    with st.chat_message("assistant"):
        st.markdown("Привет! 👋 Я — **Глосси**, ИИ-ассистент по Бизнес-глоссарию.")
        st.markdown("Задайте свой вопрос или выберите из самых частых:")

        # кнопки быстрых запросов — две колонки
        cols = st.columns(2)
        for i, q in enumerate(QUICK_QUESTIONS):
            with cols[i % 2]:
                if st.button(q, key=f"quick_{i}", use_container_width=True):
                    st.session_state.pending_question = q
                    st.rerun()

        st.caption("ℹ️ Что я умею и не умею — во вкладке **«О Глосси»**")

    # история диалога
    last_user_idx = None
    for idx, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            last_user_idx = idx

    for idx, msg in enumerate(st.session_state.messages):
        if idx == last_user_idx and st.session_state.scroll_to_last_question:
            st.markdown('<div id="last-question"></div>', unsafe_allow_html=True)

        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_assistant_message(
                    content    = msg["content"],
                    log_id     = msg.get("log_id"),
                    avg_score  = msg.get("avg_score", 0.0),
                    no_answer  = msg.get("no_answer", False),
                    next_step  = msg.get("next_step", False),
                )
            else:
                st.markdown(msg["content"])

    if st.session_state.scroll_to_last_question:
        scroll_to_last_question()
        st.session_state.scroll_to_last_question = False

    # обработка pending_question от кнопок
    if st.session_state.pending_question:
        q = st.session_state.pending_question
        st.session_state.pending_question = None
        process_question(q, vectorstore, api_key)


def process_question(question, vectorstore, api_key):
    if not api_key:
        st.error("Нет API ключа.")
        return

    # Сразу добавляем вопрос в историю и показываем его
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Теперь думаем и показываем ответ
    with st.chat_message("assistant"):
        with st.spinner("Глосси думает…"):
            try:
                answer, docs, scores, avg_score, no_answer, next_step, topic = rag_answer(
                    question, vectorstore, api_key
                )
            except Exception as e:
                answer    = f"Ошибка: {e}"
                docs, scores, avg_score, no_answer, next_step, topic = [], [], 0.0, False, False, "Другое"

        # Рендерим ответ сразу внутри chat_message("assistant")
        render_assistant_message(
            content   = answer,
            log_id    = None,   # log_id ещё не известен, лайки появятся после rerun
            avg_score = float(avg_score),
            no_answer = no_answer,
            next_step = next_step,
        )

    sources_payload = [
        {
            "topic"  : d.metadata.get("topic", ""),
            "file"   : d.metadata.get("source_file", ""),
            "score"  : float(scores[i]) if i < len(scores) else None,
            "snippet": d.page_content[:150],
        }
        for i, d in enumerate(docs)
    ]

    log_id = db_insert_log(question, answer, float(avg_score), no_answer, sources_payload, topic)
    update_session_metrics(no_answer, avg_score)

    st.session_state.messages.append({
        "role"      : "assistant",
        "content"   : answer,
        "log_id"    : log_id,
        "avg_score" : float(avg_score),
        "no_answer" : no_answer,
        "next_step" : next_step,
        "topic"     : topic,
        "feedback"  : None,
    })

    if next_step:
        st.session_state.next_step_mode = True
    else:
        st.session_state.next_step_mode = False

    db_load_logs.clear(); db_load_metrics.clear()
    st.rerun()


# ═══════════════════════════════════════════════════════════════
# СТРАНИЦА — О ГЛОССИ (заменяет старую page_about целиком)
# ═══════════════════════════════════════════════════════════════
def page_about():
    st.markdown("## 🤖 О Глосси")
 
    # ── Видео-представление ──
    st.markdown("""
    <div style="
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    ">
        <video
            width="100%"
            controls
            controlsList="nodownload"
            poster="https://raw.githubusercontent.com/aiaiai-hi/assistant/main/assets/glossy_intro.mp4"
            style="display: block; border-radius: 12px;"
        >
            <source src="{video_url}" type="video/mp4">
            Ваш браузер не поддерживает воспроизведение видео.
        </video>
    </div>
    """.format(video_url=GLOSSY_VIDEO_URL), unsafe_allow_html=True)
 
    st.markdown("""
**Глосси** — ИИ-ассистент по работе с Бизнес-глоссарием Банка.
Помогает разобраться в процессах управления отчётным ландшафтом,
отвечает на вопросы и проводит по шагам регистрации отчётов.
    """)
 
    st.markdown("---")
    col1, col2 = st.columns(2)
 
    with col1:
        st.markdown("### ✅ Что умею")
        st.markdown("""
1. Отвечать на общие вопросы по отчётам и БГ
2. Рассказать про типы запросов и отчётов
3. Провести по процессу верхнеуровнево или пошагово
4. Объяснить конкретный шаг подробно
5. Помочь разобраться с трудностями при описании в БГ
6. Отвечать на вопросы по управлению отчётным ландшафтом
        """)
 
    with col2:
        st.markdown("### ❌ Что пока не умею")
        st.markdown("""
- Давать информацию по конкретным отчётам
- Генерировать атрибутный состав
  → раздел **«Сформировать атрибуты»**
- Создавать карточки запросов
        """)
 
    st.markdown("---")
    st.markdown("### 💬 Частые вопросы")
    for ex in QUICK_QUESTIONS:
        st.markdown(f"> 💬 *{ex}*")
 
    st.markdown("---")
    st.info(
        "Не нашли ответ? Обратитесь к команде **Управления отчётным ландшафтом** "
        "или **Аналитики данных** — контакты в разделе «Контакты»."
    )
 


# ═══════════════════════════════════════════════════════════════
# СТРАНИЦА — СТАТИСТИКА
# ═══════════════════════════════════════════════════════════════
def page_stats():
    import pandas as pd
    from collections import Counter

    st.markdown("## 📊 Статистика — все сессии")

    logs    = db_load_logs()
    metrics = db_load_metrics()
    total   = metrics["total"]

    st.markdown("### Текущая сессия")
    m     = st.session_state.session_metrics
    t     = m["total"] or 1
    lk_p  = round(m["likes"]     / t * 100)
    no_p  = round(m["no_answer"] / t * 100)
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card"><div class="val">{m['total']}</div><div class="lbl">Вопросов</div></div>
        <div class="metric-card"><div class="val">👍 {m['likes']}</div><div class="lbl">Помогло ({lk_p}%)</div></div>
        <div class="metric-card"><div class="val">👎 {m['dislikes']}</div><div class="lbl">Не помогло</div></div>
        <div class="metric-card"><div class="val">{no_p}%</div><div class="lbl">«Не знаю»</div></div>
        <div class="metric-card"><div class="val">{m['avg_score']:.2f}</div><div class="lbl">Avg score</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Все сессии (из БД)")
    if total == 0:
        st.info("Пока вопросов не было. Перейдите в «Чат» и задайте первый вопрос!")
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Всего вопросов", total)
    c2.metric("👍 Помогло",     metrics["likes"],
              f"{round(metrics['likes']/total*100)}%")
    c3.metric("👎 Не помогло",  metrics["dislikes"],
              f"{round(metrics['dislikes']/total*100)}%")
    c4.metric("⚠️ «Не знаю»",  metrics["no_answer"],
              f"{round(metrics['no_answer']/total*100)}%")
    c5.metric("Avg retrieval",  metrics["avg_score"])

    st.markdown("---")

    # Распределение по темам
    st.markdown("### 🏷️ Частые темы запросов")
    topics_in_session = [
        m.get("topic", "Другое")
        for m in st.session_state.messages
        if m["role"] == "assistant" and m.get("topic")
    ]
    if topics_in_session:
        top_topics = Counter(topics_in_session).most_common()
        df_topics  = pd.DataFrame(top_topics, columns=["Тема", "Кол-во"])
        st.dataframe(df_topics, use_container_width=True, hide_index=True)
    else:
        st.caption("Темы появятся после первых вопросов в этой сессии.")

    st.markdown("---")
    st.markdown("### 📈 Релевантность поиска по вопросам")
    if len(logs) >= 2:
        df_sc = pd.DataFrame({
            "№"    : range(1, len(logs) + 1),
            "Score": [float(r["avg_score"] or 0) for r in logs],
        }).set_index("№")
        st.line_chart(df_sc)
    else:
        st.caption("Нужно минимум 2 вопроса для графика.")

    st.markdown("---")
    st.markdown("### 📄 Топ источников")
    all_sources = []
    for rec in logs:
        for src in (rec.get("sources") or []):
            label = src.get("topic") or src.get("file") or "—"
            all_sources.append(label)
    if all_sources:
        top    = Counter(all_sources).most_common(10)
        df_top = pd.DataFrame(top, columns=["Источник", "Раз использован"])
        st.dataframe(df_top, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🗂️ Все вопросы")
    fb_filter = st.selectbox(
        "Фильтр по оценке:",
        ["Все", "👍 Помогло", "👎 Не помогло", "Без оценки"],
        key="fb_filter",
    )
    filter_map = {"Все": None, "👍 Помогло": "like",
                  "👎 Не помогло": "dislike", "Без оценки": "none"}
    selected = filter_map[fb_filter]

    shown = 0
    for rec in reversed(logs):
        fb = rec["feedback"]
        if selected == "none"             and fb is not None:  continue
        if selected in ("like","dislike") and fb != selected:  continue

        fb_html = (
            '<span class="fb-like">👍 Помогло</span>'       if fb == "like"    else
            '<span class="fb-dislike">👎 Не помогло</span>' if fb == "dislike" else
            '<span class="fb-none">— без оценки</span>'
        )
        no_tag = ' · <span style="color:#f59e0b">⚠️ «Не знаю»</span>' if rec["no_answer"] else ""
        ts     = (rec.get("created_at") or "")[:16].replace("T", " ")

        with st.expander(
            f"#{rec['id']}  {ts}  ·  score {float(rec['avg_score'] or 0):.2f}",
            expanded=False,
        ):
            st.markdown(f"**Вопрос:** {rec['question']}")
            st.markdown(f"**Ответ:** {rec['answer']}")
            st.markdown(f"**Оценка:** {fb_html}{no_tag}", unsafe_allow_html=True)
            if rec.get("sources"):
                st.markdown("**Источники:**")
                for s in rec["sources"]:
                    bar = int((s.get("score") or 0) * 100)
                    st.markdown(
                        f"- `{s.get('topic','—')}` score {s.get('score')}  "
                        f'<span style="display:inline-block;width:80px;vertical-align:middle">'
                        f'<div class="score-bar-wrap">'
                        f'<div class="score-bar" style="width:{bar}%"></div>'
                        f'</div></span>',
                        unsafe_allow_html=True,
                    )
        shown += 1

    if shown == 0:
        st.info("Нет записей по выбранному фильтру.")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="Глосси — ИИ-ассистент",
        page_icon="🤖",
        layout="wide",
    )
    inject_styles()
    init_state()

    # хедер с картинкой из репозитория
    st.markdown(f"""
    <div class="glossy-header">
        <img class="glossy-header-img" src="{GLOSSY_IMG_URL}" alt="Глосси"/>
        <div class="glossy-header-text">
            <h1>Глосси</h1>
            <p class="subtitle">ИИ-ассистент по Бизнес-Глоссарию</p>
            <p class="desc">Задайте вопрос — получите ответ из базы знаний</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        api_key = st.secrets["QWEN_API_KEY"]
    except Exception:
        api_key = None
        st.warning("⚠️ Добавьте QWEN_API_KEY в `.streamlit/secrets.toml`")

    vectorstore = None
    try:
        vectorstore = load_vectorstore()
    except Exception as e:
        st.error(f"❌ Не удалось загрузить FAISS-индекс: {e}")

    render_nav()

    active = st.session_state.active_tab

    if active == "chat":
        if vectorstore:
            page_chat(vectorstore, api_key)
        else:
            st.info("Индекс не загружен — убедитесь, что папка `faiss_index` рядом с `app.py`.")
    elif active == "about":
        page_about()
    elif active == "stats":
        page_stats()

    # chat_input — закреплён внизу страницы
    if active == "chat" and vectorstore:
        question = st.chat_input("Задайте вопрос по Бизнес-Глоссарию…")
        if question:
            process_question(question, vectorstore, api_key)

    with st.sidebar:
        st.markdown("### 🤖 Глосси v10")
        st.caption("RAG · FAISS · Qwen · Supabase")
        st.markdown("---")
        m = st.session_state.session_metrics
        st.caption(f"Вопросов в сессии: **{m['total']}**")
        st.caption(f"👍 {m['likes']}  👎 {m['dislikes']}")
        if st.session_state.next_step_mode:
            st.info("📖 Пошаговый режим активен")
        st.markdown("---")
        if st.button("🗑️ Очистить чат", use_container_width=True):
            st.session_state.pop("messages", None)
            st.session_state.pop("session_metrics", None)
            st.session_state.pop("scroll_to_last_question", None)
            st.session_state.pop("next_step_mode", None)
            st.session_state.pop("pending_question", None)
            st.rerun()


if __name__ == "__main__" or True:
    main()
