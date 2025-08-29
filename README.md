# RAG-X-Quan
Momentum Strategy with Local RAG — 將量化動能策略與在地檢索增強生成 (RAG) 結合，融合台灣市場研究與國際文獻，建立可回測、解釋性強的 ETF/股票投資框架。

# Momentum Strategy with Local RAG

# Local RAG Pipeline

這個專案提供一個 **在地 RAG (Retrieval-Augmented Generation)** 的最小可行實作，支援本地 PDF / TXT / Markdown 檔案，並透過 **FAISS 向量資料庫 + Ollama 模型** 完成檢索與問答。主要功能包括：📂 多格式文件支援 (PDF、TXT、Markdown)、🔎 文件切分 (`RecursiveCharacterTextSplitter`)，確保檢索單位合理、🧠 在地模型推理 (Embedding 預設 `bge-m3:latest`，可換 `nomic-embed-text`；Chat 預設 `gpt-oss:20b`，可改成本地 Ollama 模型)、💾 向量儲存 (使用 FAISS 建立/更新索引)、⚡ 快速檢索 (從本地文件找到相關段落，再交給 ChatOllama 回答)。  


**為什麼不直接向大型LLM提問？**
1.	貼近台灣市場特性
	國外研究多半以美股、ETF 為基準，但台股/台灣 ETF 的流動性、交易稅、產業結構差異很大。
	在地 RAG 可以把「台灣學術研究」和「本地投資實證」拉進來，避免直接照搬美國模型失真。
2.	結合最新資料與本地知識
	台股有獨特因子（融資融券、法人買賣超、除權息、集中度），這些在國外資料庫找不到。
	RAG 能即時檢索到台灣券商/學術論文/市場報告，讓策略參數（持有期、篩選條件）更符合本地實況。
3.	避免黑箱，強化可解釋性
	傳統策略可能給你一個「12 個月回報 → Top10 → 均線過濾」就結束。
	RAG 可以讓你每一步都有「依據哪篇論文/哪份報告」→ 更容易在申請部門、報告或面試時說服別人。
	對你來說，不只是「跑出績效」，而是能拿出 台灣本地研究+數據支持。
4.	彈性調整
	傳統架構 → 規則死板。
	在地 RAG → 可以動態查詢不同市場環境下的最佳持有期、排名方式、風控條件，讓策略不會被單一市場 regime 卡死。
---

## 使用流程
1. **準備環境**  
   安裝必要套件：  
   ```bash
   pip install langchain langchain-community langchain-ollama faiss-cpu pypdf pymupdf
2. **放置文件**
    將 PDF/TXT/Markdown 檔案放到 RAG_DATA_DIR 資料夾。
	或自行指定路徑： export RAG_DATA_DIR=/path/to/your/docs
3. **建立索引**
    第一次執行程式時，會自動將文件切分、建立 embedding，並存到 index/ 資料夾。
	之後新增文件再執行時，會自動更新索引。
4. **提問互動**
    在終端機輸入查詢或筆記本內提問
    給我一個超棒的動能策略
5. **專案結構**
    ├── rag_local.py       # 主程式
    ├── index/             # FAISS 索引儲存位置
    └── RAG_DATA_DIR/      # 放置 PDF/TXT/Markdown 文件
6. **環境變數**
	EMBED_MODEL：embedding 模型名稱，預設 bge-m3:latest
	CHAT_MODEL：聊天模型名稱，預設 gpt-oss:20b
	RAG_DATA_DIR：文件目錄路徑