# Building a Fine-tuned LLM + RAG Recommendation System

##  Project Overview 專案概述

This project builds a **fashion recommendation system** by combining:

- **Fine-tuned LLM** on instruction-style fashion data  
- **FAISS vector database** for efficient similarity search  
- **RAG (Retrieval-Augmented Generation)** to generate personalized fashion recommendations

**LLM fine-tuning dataset**: `neuralwork/fashion-style-instruct` (Hugging Face)  
**RAG retrieval dataset**: H&M Personalized Fashion Recommendations dataset  
(https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data?select=articles.csv)  
**Demo**: `floraliuya/recft_unsloth-Meta-Llama-3.1-8B-2` (Hugging Face)

---

##  High-level Pipeline 步驟總覽

1. **建立向量資料庫（FAISS）**  
   - 將 H&M 時尚商品與使用者互動資料轉成向量，存進 FAISS。
2. **模型微調（Fine-tune LLM）**  
   - 使用 `fashion-style-instruct` 資料集，在基礎 LLM 上做指令微調，讓模型學會理解穿搭需求與風格偏好。
3. **建構 RAG 流程（RAG for Recommendation）**  
   - 根據使用者描述或歷史行為檢索相關商品，  
   - 將檢索結果丟給微調後的 LLM，產生 **可解釋、自然語言的推薦結果**。

---

##  Step 1 – Build the FAISS Vector Store 建立向量資料庫

### 1.1 Data Preparation 資料準備

From the **H&M Personalized Fashion Recommendations** dataset:

- 商品層級資料（items）
  - 商品 ID、名稱、類別（上衣、褲子、鞋子等）
  - 顏色、材質、季節、價格區間
  - 商品描述（text）

### 1.2 Embedding & Index 向量化與索引

1. 選擇一個 **text embedding 模型**（例如 sentence-transformer 或 LLM 的 embedding head）。
2. 為每個商品建立 text representation，例如：
   - `"{product_title}. {product_description}. Category: {category}. Color: {color}."`
3. 將這些文本餵給 embedding 模型，取得向量表示。
4. 使用 **FAISS** 建立向量索引：
   - 適合中小型資料集：`IndexFlatL2`
   - 資料量大時可使用 IVF / HNSW 等加速結構

> 結果：得到一個 `FAISS index`，可以根據 文字描述 快速找出最相似的商品。

---

## Step 2 – Fine-tune the LLM 模型微調

### 2.1 Dataset 使用資料集

- **LLM will be fine-tuned on**: `fashion-style-instruct` (Hugging Face)  
- 這個資料集包含：
  - 使用者對話式需求（e.g., "I want a casual summer outfit for a weekend trip."）
  - 模型期望輸出（e.g., 推薦哪些單品、搭配建議、風格解釋）

### 2.2 Fine-tuning Objective 微調目標

讓 LLM 學會：

- 理解自然語言描述的 **穿搭需求 / 場景 / 風格關鍵字**
- 以 **指令式輸入** 產生：
  - 推薦理由（為什麼推薦這些單品）
  - 風格說明（e.g., 「這套穿搭偏休閒、適合戶外」）
  - 個人化調整建議（例如「如果你不喜歡亮色，可以換成深藍色上衣」）

### 2.3 Expected Input/Output 格式示例

- **Input (Instruction)**  
  > "I’m looking for a yellow summer dress which is light and airy"

- **Output (Model Response)**  
  - 列出 2–4 件單品（上衣、褲子、鞋子、配件）  
  > "1. Covent Garden — Dress — Light Yellow — Ladieswear — Sleeveless maxi dress in crinkled viscose with inset lace sections,crocheted lace shoulder straps and a seam with flounces at the hem. Unlined.
  > 2. SUMMER STRAP DRESS— Dress — Light Yellow — Divided — Short dress in soft jersey with a V-neck"

---

## Step 3 – RAG for Fashion Recommendation 使用 RAG 做推薦

### 3.1 RAG Core Idea 核心概念

RAG combines:

1. **Retriever（檢索器）**  
   - 使用 FAISS 向量庫，根據使用者需求或 profile 搜尋 **候選商品集合**。
2. **Generator（生成器，微調後的 LLM）**  
   - 將檢索到的商品資訊（如標題、描述、顏色、類別）  
     以 context 的形式餵給 LLM。  
   - LLM 根據這些 context 產生：
     - 個人化推薦清單
     - 自然語言解釋
     - 風格搭配建議

### 3.2 RAG Inference Flow 推論流程

1. **User Query / Profile Input 使用者輸入**
   - 例：  
     > "I’m looking for a formal shirt made of linen"

2. **Encode & Retrieve 編碼與檢索**
   - 將使用者輸入轉成向量  
   - 在 FAISS 中搜尋 top-k 相似的商品（例如 k=20）

3. **Build Context 建立上下文**
   - 將檢索到的商品資訊整理成可讀文本，例如：
     - 商品名稱 / 風格 / 價格 / 類別 / 顏色
   - 放入一個 prompt template，給 LLM 使用

4. **LLM Generation 生成推薦**
   - 使用微調後的 LLM，從 context 中挑選適合的單品組合
   - 產生回答內容，例如：
     - 推薦哪些商品（以 H&M 商品 ID 關聯）
     - 為什麼挑這些（根據顏色、版型、季節、風格）
     - 如有需要，給出多套備選穿搭方案

---

##  Future Extensions 可能延伸方向

- 加入 **使用者歷史行為**（購買紀錄 / 點擊紀錄）做個人化 re-ranking  
- 加入 **多模態資訊**：圖片 embedding + 文字描述一起放入 FAISS  
- 使用 **評估指標**（如 CTR、轉換率、NDCG）來量化推薦效益  
- 加入 A/B test 或 offline evaluation pipeline

---

