import os
import re
import glob
import hashlib

import streamlit as st

from langchain_community.document_loaders import (
    TextLoader,
    PDFPlumberLoader,
)

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# 1. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PDPA Legal Assistant",
    page_icon="⚖️",
    layout="wide",
)


# ============================================================
# 2. TITLE
# ============================================================

st.title("⚖️ PDPA Legal Assistant")
st.caption(
    "ระบบผู้ช่วยตอบคำถามเกี่ยวกับพระราชบัญญัติคุ้มครองข้อมูลส่วนบุคคล "
    "พ.ศ. 2562 ด้วยเทคนิค Retrieval-Augmented Generation (RAG)"
)


# ============================================================
# 3. PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


# ============================================================
# 4. GEMINI API KEY
# ============================================================

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error(
        "ไม่พบ GEMINI_API_KEY\n\n"
        "กรุณาตั้งค่าใน .streamlit/secrets.toml หรือตั้งค่า Environment Variable"
    )
    st.stop()


# ============================================================
# 5. FILE SIGNATURE
# ============================================================

def get_data_signature():
    files = []
    files.extend(glob.glob(os.path.join(DATA_DIR, "*.txt")))
    files.extend(glob.glob(os.path.join(DATA_DIR, "*.pdf")))
    files = sorted(files)

    information = []
    for file_path in files:
        try:
            stat = os.stat(file_path)
            information.append(
                (
                    os.path.basename(file_path),
                    stat.st_size,
                    stat.st_mtime_ns,
                )
            )
        except OSError:
            pass

    raw = repr(information).encode("utf-8")
    return hashlib.md5(raw).hexdigest()


# ============================================================
# 6. TEXT CLEANING
# ============================================================

def clean_text(text):
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ============================================================
# 7. LOAD TXT FILES
# ============================================================

def load_txt_files():
    documents = []
    txt_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))

    for file_path in txt_files:
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()

            for doc in docs:
                text = clean_text(doc.page_content)
                if not text:
                    continue

                doc.page_content = text
                doc.metadata["source"] = os.path.basename(file_path)
                documents.append(doc)

        except Exception as e:
            st.warning(f"อ่านไฟล์ไม่ได้: {os.path.basename(file_path)}\n{e}")

    return documents


# ============================================================
# 8. LOAD PDF FILES (ปรับแก้ไขให้โหลด PDF ได้ทุกไฟล์ในโฟลเดอร์)
# ============================================================

def load_pdf_files():
    documents = []
    pdf_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))

    for file_path in pdf_files:
        try:
            loader = PDFPlumberLoader(file_path)
            docs = loader.load()

            for doc in docs:
                text = clean_text(doc.page_content)
                if not text:
                    continue

                doc.page_content = text
                doc.metadata["source"] = os.path.basename(file_path)
                documents.append(doc)

        except Exception as e:
            st.warning(f"ไม่สามารถอ่าน PDF ได้: {os.path.basename(file_path)}\n{e}")

    return documents


# ============================================================
# 9. THAI NUMBER CONVERSION
# ============================================================

THAI_TO_ARABIC = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")


def thai_to_arabic(text):
    return text.translate(THAI_TO_ARABIC)


# ============================================================
# 10. DETECT ARTICLE
# ============================================================

def detect_article_number(question):
    pattern = re.compile(r"มาตรา\s*([๐-๙0-9]+)")
    match = pattern.search(question)

    if not match:
        return None

    try:
        return int(thai_to_arabic(match.group(1)))
    except Exception:
        return None


# ============================================================
# 11. SPLIT BY ARTICLE
# ============================================================

def split_articles(documents):
    article_documents = []
    pattern = re.compile(r"(?m)^\s*มาตรา\s+([๐-๙0-9]+)")

    for document in documents:
        text = document.page_content
        matches = list(pattern.finditer(text))

        if not matches:
            article_documents.append(document)
            continue

        # เก็บเนื้อหาส่วนหัว (บทนำ / ชื่อ พ.ร.บ. / คำปรารภ) ก่อนถึงมาตราแรก
        if matches[0].start() > 0:
            header_text = text[:matches[0].start()].strip()
            if header_text:
                metadata = dict(document.metadata)
                metadata["article_title"] = "บทนำ / ภาพรวม"
                article_documents.append(
                    Document(page_content=header_text, metadata=metadata)
                )

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            article_text = text[start:end].strip()
            if not article_text:
                continue

            article_number = int(thai_to_arabic(match.group(1)))

            metadata = dict(document.metadata)
            metadata["article"] = article_number
            metadata["article_title"] = f"มาตรา {article_number}"

            article_documents.append(
                Document(page_content=article_text, metadata=metadata)
            )

    return article_documents


# ============================================================
# 12. CHUNKING
# ============================================================

def create_chunks(documents):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    return chunks


# ============================================================
# 13. BUILD VECTOR DATABASE
# ============================================================

@st.cache_resource
def build_vectorstore(data_signature):
    txt_documents = load_txt_files()
    pdf_documents = load_pdf_files()  # ใช้ฟังก์ชันที่ปรับปรุงใหม่
    all_documents = txt_documents + pdf_documents

    article_documents = split_articles(all_documents)

    if article_documents:
        documents_for_rag = article_documents
    else:
        documents_for_rag = all_documents

    chunks = create_chunks(documents_for_rag)

    vectorstore = None
    if chunks:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        vectorstore = FAISS.from_documents(chunks, embeddings)

    return {
        "documents": all_documents,
        "articles": article_documents,
        "chunks": chunks,
        "vectorstore": vectorstore,
    }


# ============================================================
# 14. BUILD RAG
# ============================================================

data_signature = get_data_signature()
rag = build_vectorstore(data_signature)

all_documents = rag["documents"]
article_documents = rag["articles"]
chunks = rag["chunks"]
vectorstore = rag["vectorstore"]


# ============================================================
# 15. SIDEBAR
# ============================================================

with st.sidebar:
    st.header("🔧 RAG Information")
    st.write(f"📄 Documents: **{len(all_documents)}**")
    st.write(f"⚖️ Article Documents: **{len(article_documents)}**")
    st.write(f"🧩 Chunks: **{len(chunks)}**")

    if vectorstore:
        st.success("FAISS พร้อมใช้งาน")
    else:
        st.error("ไม่พบ Vector Database")

    st.divider()
    st.write("**Data Directory**")
    st.code(DATA_DIR)
    st.divider()

    if st.button("🔄 โหลดเอกสารใหม่"):
        st.cache_resource.clear()
        st.rerun()


# ============================================================
# 16. GEMINI (เปลี่ยนเป็นชื่อโมเดลมาตรฐาน gemini-1.5-flash)
# ============================================================


llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",  # ✅ เปลี่ยนกลับเป็น gemini-3.8-flash ตามที่ API แนะนำ
    temperature=0,
    google_api_key=GEMINI_API_KEY,
)


# ============================================================
# 17. PROMPT
# ============================================================

SYSTEM_PROMPT = """
คุณคือ PDPA Legal Assistant
สำหรับตอบคำถามเกี่ยวกับพระราชบัญญัติคุ้มครองข้อมูลส่วนบุคคล
พ.ศ. 2562 จากเอกสารที่ระบบค้นพบ

กฎในการตอบ:

1. ตอบจาก CONTEXT เท่านั้น
2. ห้ามใช้ความรู้ภายนอก CONTEXT
3. ห้ามแต่งข้อมูลขึ้นเอง
4. หาก CONTEXT ไม่มีข้อมูลที่เพียงพอสำหรับตอบคำถาม
   ให้ตอบว่า "ไม่พบข้อมูล"
5. หากคำถามระบุเลขมาตรา
   ให้ตอบโดยอ้างอิงเนื้อหาของมาตรานั้นเป็นหลัก
6. หากตอบได้ ให้ระบุเลขมาตราที่เกี่ยวข้อง
7. ใช้ภาษาไทยที่เข้าใจง่าย
8. ไม่ต้องกล่าวว่าเป็น AI
9. ไม่ต้องสร้างข้อกฎหมายที่ไม่มีอยู่ใน CONTEXT
10. ให้ถือว่าคำว่า "PDPA" หมายถึง "พระราชบัญญัติคุ้มครองข้อมูลส่วนบุคคล พ.ศ. 2562"

CONTEXT
==================================================

{context}

==================================================

QUESTION

{question}

==================================================

ANSWER
"""


# ============================================================
# 18. DIRECT ARTICLE RETRIEVAL
# ============================================================

def get_article(article_number):
    if not article_documents:
        return None

    for document in article_documents:
        if document.metadata.get("article") == article_number:
            return document

    return None


# ============================================================
# 19. RETRIEVAL
# ============================================================

def retrieve_documents(question):
    article_number = detect_article_number(question)

    if article_number is not None:
        article_doc = get_article(article_number)
        if article_doc:
            return [article_doc], "article"
        return [], "article_not_found"

    if vectorstore is None:
        return [], "no_vectorstore"

    try:
        results = vectorstore.similarity_search(question, k=5)
        return results, "semantic"

    except Exception as e:
        st.error(f"Retrieval error: {e}")
        return [], "error"


# ============================================================
# 20. CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# 21. CHAT INPUT
# ============================================================

question = st.chat_input(
    "ลองถาม เช่น PDPA คืออะไร? หรือ มาตรา 27 กำหนดเกี่ยวกับอะไร?"
)


if question:
    # แสดงคำถามเดิมของผู้ใช้บน UI
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # แปลงคำว่า PDPA เป็นชื่อเต็มเพื่อส่งให้ Search และ LLM เข้าใจตรงกัน
    processed_question = re.sub(
        r"\bPDPA\b",
        "พระราชบัญญัติคุ้มครองข้อมูลส่วนบุคคล (PDPA)",
        question,
        flags=re.IGNORECASE,
    )

    with st.spinner("⚖️ กำลังค้นหาข้อกฎหมายและประมวลผลคำตอบ..."):
        retrieved_docs, mode = retrieve_documents(processed_question)
        article_number = detect_article_number(processed_question)

        if not retrieved_docs:
            answer = "ไม่พบข้อมูล"
            context = ""
        else:
            context_parts = []
            sources = []

            for i, document in enumerate(retrieved_docs, start=1):
                article = document.metadata.get("article")
                source = document.metadata.get("source", "ไม่ทราบแหล่งข้อมูล")

                if article:
                    source_label = f"{source} (มาตรา {article})"
                else:
                    source_label = source

                context_parts.append(
                    f"""
[Context {i}]
Source: {source_label}

{document.page_content}
"""
                )
                sources.append(source_label)

            context = "\n\n".join(context_parts)
            prompt = SYSTEM_PROMPT.format(
                context=context, 
                question=processed_question
            )

            # เพิ่ม Error Handling ดักจับ Quota Limit
            try:
                response = llm.invoke(prompt)

                if isinstance(response.content, list):
                    text_parts = []
                    for part in response.content:
                        if isinstance(part, dict) and "text" in part:
                            text_parts.append(part["text"])
                        elif isinstance(part, str):
                            text_parts.append(part)
                    answer = "".join(text_parts)
                else:
                    answer = str(response.content)

                if not answer.strip():
                    answer = "ไม่พบข้อมูล"

            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    answer = "⚠️ **โควตาการใช้งาน API เต็มชั่วคราว (Rate Limit / Quota Exceeded)**\n\nกรุณาเปลี่ยน Gemini API Key หรือเว้นระยะเวลาสักครู่แล้วลองใหม่อีกครั้ง"
                else:
                    answer = f"เกิดข้อผิดพลาดในการเรียก LLM:\n\n{err_msg}"

    with st.chat_message("assistant"):
        with st.expander("🔍 ตรวจสอบ Retrieval", expanded=False):
            if article_number:
                st.write(f"🎯 ตรวจพบมาตรา: **{article_number}**")
                if mode == "article":
                    st.success("ใช้ Article-based Retrieval")
                else:
                    st.warning("ไม่พบมาตรานี้ในเอกสาร")
            else:
                st.write("🔎 ใช้ Semantic Vector Search")

            st.write(f"พบ Context: **{len(retrieved_docs)}** รายการ")

        if retrieved_docs and context:
            with st.expander("📄 Context ที่ส่งให้ LLM"):
                st.text(context)

        st.markdown(answer)

        if retrieved_docs:
            st.divider()
            st.markdown("### 📚 แหล่งข้อมูล")
            displayed_sources = set()

            for document in retrieved_docs:
                source = document.metadata.get("source", "ไม่ทราบแหล่งข้อมูล")
                article = document.metadata.get("article")

                if article:
                    label = f"📄 {source} — มาตรา {article}"
                else:
                    label = f"📄 {source}"

                if label not in displayed_sources:
                    st.caption(label)
                    displayed_sources.add(label)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )