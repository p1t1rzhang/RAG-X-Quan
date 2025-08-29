# rag_local.py
from pathlib import Path
from typing import List, Tuple
import os
import sys

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# 可能用到的 PDF loader（有就用、沒有就跳過）
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader  # 基本款
try:
    from langchain_community.document_loaders import PyMuPDFLoader  # 備援（fitz）
    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False

INDEX_DIR = Path("index")
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", "/Users/p1t1rzhang/Desktop/rag_local/RAG_DATA_DIR"))

# 預設 embedding/聊天模型；可用環境變數覆蓋
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3:latest")  # 若沒拉過，可改成 "nomic-embed-text"
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-oss:20b")  # 先用你本機現成的；之後可切 gpt-oss:20b


def is_pdf(path: Path) -> bool:
    return path.suffix.lower() == ".pdf"


def is_txt(path: Path) -> bool:
    return path.suffix.lower() in {".txt", ".md", ".markdown"}


def looks_like_pdf_binary(path: Path) -> bool:
    # 快速檢查檔頭是否為 %PDF-
    try:
        with open(path, "rb") as f:
            head = f.read(5)
        return head.startswith(b"%PDF-")
    except Exception:
        return False


def load_one_file(path: Path) -> List:
    """逐檔載入；PDF 先用 PyPDFLoader，失敗再用 PyMuPDFLoader（若可用）。"""
    docs = []
    try:
        if is_txt(path):
            docs.extend(TextLoader(str(path), encoding="utf-8").load())
        elif is_pdf(path):
            if not looks_like_pdf_binary(path):
                print(f"⚠️  跳過（不是合法 PDF 頭）：{path.name}")
                return docs
            try:
                docs.extend(PyPDFLoader(str(path)).load())
            except Exception as e1:
                if HAS_PYMUPDF:
                    try:
                        docs.extend(PyMuPDFLoader(str(path)).load())
                    except Exception as e2:
                        print(f"⚠️  PDF 解析失敗（pypdf & pymupdf）：{path.name} | {e2}")
                else:
                    print(f"⚠️  PDF 解析失敗（pypdf），且未安裝 pymupdf：{path.name} | {e1}")
        # 其他副檔名忽略
    except Exception as e:
        print(f"⚠️  載入失敗：{path.name} | {e}")
    return docs


def load_and_split(data_dir: Path):
    if not data_dir.exists():
        raise FileNotFoundError(f"資料夾不存在：{data_dir}")

    files = [p for p in data_dir.rglob("*") if p.is_file() and (is_pdf(p) or is_txt(p))]
    if not files:
        raise ValueError(f"在 {data_dir} 沒找到 PDF/TXT 文件")

    loaded_docs = []
    skipped = 0
    print(f"📦 準備載入 {len(files)} 個檔案…")
    for p in sorted(files):
        docs = load_one_file(p)
        if docs:
            loaded_docs.extend(docs)
            print(f"  ✅ {p.name} -> {len(docs)} docs")
        else:
            skipped += 1
            print(f"  ⏭️  跳過：{p.name}")

    print(f"📊 載入完成：成功 {len(loaded_docs)}，跳過 {skipped}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150, add_start_index=True)
    chunks = splitter.split_documents(loaded_docs)
    print(f"✂️  切塊完成：{len(chunks)} chunks")
    return chunks


def build_or_load_index(chunks):
    embed = OllamaEmbeddings(model=EMBED_MODEL)
    idx_path = INDEX_DIR / "faiss_index"

    faiss_bin = idx_path.with_suffix(".faiss")
    faiss_meta = idx_path.with_suffix(".pkl")
    if faiss_bin.exists() and faiss_meta.exists():
        vs = FAISS.load_local(folder_path=str(idx_path), embeddings=embed, allow_dangerous_deserialization=True)
    else:
        os.makedirs(INDEX_DIR, exist_ok=True)
        vs = FAISS.from_documents(chunks, embed)
        vs.save_local(str(idx_path))
    return vs


def answer_question(vs, question: str, k: int = 5) -> Tuple[str, List[Tuple[str, str]]]:
    # 使用 MMR 檢索降低重複
    retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": k, "lambda_mult": 0.3})
    # 關鍵修正：傳入純字串，不要 dict
    try:
        hits = retriever.invoke(question)
    except TypeError:
        # 相容舊法
        hits = retriever.get_relevant_documents(question)

    if not hits:
        msg = ("【沒有檢索到內容】\n"
               "請確認資料夾內有可讀的 TXT/無密碼 PDF；或換個關鍵字再問一次。")
        return msg, []

    # 收集來源與 context
    sources: List[Tuple[str, str]] = []
    context_blocks: List[str] = []
    for d in hits:
        meta = d.metadata or {}
        fname = Path(meta.get("source", "unknown")).name
        page = f"p.{meta.get('page', '')}" if meta.get("page") is not None else ""
        sources.append((fname, page))
        context_blocks.append(d.page_content)

    context_text = "\n\n---\n\n".join(context_blocks)

    llm = ChatOllama(model=CHAT_MODEL, temperature=0.2)
    system_msg = (
        "你是量化研究助理。請嚴格根據提供的【檢索內容】回答；"
        "若文件未提及，就說不知道，避免幻覺。回覆末尾不需要再次列出來源。"
    )
    user_msg = f"""我的問題：{question}

【檢索內容】
{context_text}
"""

    try:
        resp = llm.invoke([("system", system_msg), ("user", user_msg)])
        answer = resp.content if hasattr(resp, "content") else str(resp)
        return answer.strip(), sources
    except Exception as e:
        err = str(e)
        tips = (f"\n\n❌ 呼叫模型失敗：{err}\n"
                f"目前 CHAT_MODEL 設定為：{CHAT_MODEL}\n"
                "請先以 `ollama list` 確認是否已下載；\n"
                "若沒有該模型：`ollama pull llama3.2:latest` 或改用 `export CHAT_MODEL=\"gpt-oss:20b\"`（拉好後再用）。")
        return tips, []


if __name__ == "__main__":
    print(f"🔧 CHAT_MODEL = {CHAT_MODEL} | EMBED_MODEL = {EMBED_MODEL}")
    print("🔍 準備建立/載入索引 ...")
    chunks = load_and_split(DATA_DIR)
    vs = build_or_load_index(chunks)

    print("✅ 就緒，開始提問（輸入空白行離開）")
    while True:
        q = input("\n問題> ").strip()
        if not q:
            break
        ans, srcs = answer_question(vs, q, k=5)
        print("\n--- 回答 ---")
        print(ans)
        if srcs:
            print("\n(參考來源)", "、".join([f"{s[0]}{(' '+s[1]) if s[1] else ''}" for s in srcs]))