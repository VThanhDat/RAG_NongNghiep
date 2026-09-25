# RAG Nông Nghiệp

Hệ thống RAG cho tài liệu nông nghiệp dạng PDF. Project hiện được tách theo hướng dễ mở rộng:

```text
Core RAG pipeline  ->  FastAPI API  ->  ReactJS chatbot frontend
```

## Kiến Trúc

```text
loader/              Đọc PDF text-layer
chunking/            Recursive chunking
embedding/           Cohere Embed v4 + BM25
vector_db/           Pinecone serverless
pre_retrieval/       Làm sạch và hiểu câu hỏi
retrieval/           Hybrid retrieval
post_retrieval/      Lọc context nhẹ
prompt/              Citation prompt
generation/          Gemini Flash generator
pipeline/            Ghép stage indexing và generation
api/                 FastAPI routes, schemas và services
frontend/            ReactJS chatbot
```

## Pipeline

Stage 1 - Indexing:

```text
PDF -> Loader -> Recursive Chunking -> Cohere Embedding API -> BM25 -> Pinecone
```

Stage 2 - Generation:

```text
Question -> Pre-retrieval -> Hybrid Retrieval -> Post-retrieval -> Citation Prompt -> Gemini Flash
```

Stack chính:

- Backend: FastAPI
- Frontend: ReactJS + Vite
- Embedding: Cohere Embed API (`embed-v4.0`, 1024 dimensions)
- Vector DB: Pinecone serverless
- LLM: Gemini 3.6 Flash API (`gemini-3.6-flash`)

> Cấu hình production/demo hiện tại dùng Cohere Embedding + Pinecone + Gemini Flash. Xem
> [DEPLOYMENT.md](DEPLOYMENT.md) để cài API key và chạy đúng flow mới.

## Cài Backend

Yêu cầu:

- Python 3.11+
- API key Cohere để tạo embedding
- API key Gemini để sinh câu trả lời bằng Gemini Flash

Tạo môi trường ảo:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Cài dependency Python:

```powershell
pip install -r requirements.txt
```

### Cấu hình Cohere Embedding

Thêm key Cohere vào `.env`:

```text
COHERE_API_KEY=your_cohere_api_key
```

Project dùng model `embed-v4.0`, vector 1024 chiều và cache local tại:

```text
.cache/cohere_embeddings.json
```

Sau khi đổi provider hoặc model, cần index lại toàn bộ tài liệu vì Pinecone không thể trộn vector khác dimension.

Chạy FastAPI API:

```powershell
uvicorn api.app.main:app --reload --host 127.0.0.1 --port 8000
```

API chính:

```text
GET  /health
POST /indexing/upload
POST /indexing/run
POST /indexing/load
GET  /indexing/status
POST /chat
```

## Cài Frontend

Mở terminal khác:

```powershell
cd frontend
npm install
npm run dev
```

Frontend mặc định gọi API tại:

```text
http://localhost:8000
```

Nếu muốn đổi API URL, copy `frontend/.env.example` thành `frontend/.env`:

```text
VITE_API_BASE_URL=http://localhost:8000
```

Sau đó mở:

```text
http://localhost:5173
```

## Cách Dùng

1. Cấu hình `COHERE_API_KEY`, `GEMINI_API_KEY` và `PINECONE_API_KEY`.
2. Chạy FastAPI API.
3. Chạy React frontend.
4. Đưa PDF vào hệ thống (xem mục **Nhóm Tri Thức** bên dưới).
5. Bấm `Index`.
6. Nhập câu hỏi trong chatbot.

Chatbot hỗ trợ **hội thoại đa lượt**: lịch sử các câu hỏi và trả lời trước đó được tự động gửi lên backend trong mỗi lượt hỏi, giúp model hiểu được ngữ cảnh của cuộc trò chuyện.

Sau khi đã index một lần, lần sau API khởi động sẽ tự thử load lại:

```text
Pinecone index: rag-nong-nghiep-cohere
Namespace: rag_collection_cohere
storage/pinecone/documents.jsonl
```

Nếu API chưa ready, có thể bấm `Load` trên frontend hoặc gọi:

```text
POST /indexing/load
```

Load lại index cũ sẽ bỏ qua bước đọc PDF, chunking và embedding lại toàn bộ tài liệu.

Nếu bạn đã index trước khi project có file `documents.jsonl`, hãy chạy `Index` lại một lần để tạo đủ artifact. Từ lần sau mới dùng được `Load`.


## Nhóm Tri Thức Bằng Metadata

Mỗi PDF/chunk có metadata `knowledge_group`. Khi chọn một nhóm cụ thể trong chatbot, retrieval chỉ tìm trong nhóm đó để giảm nhiễu và tăng tốc.

### Cơ chế hoạt động

Khi bạn chọn nhóm `lua` trong chatbot, hệ thống lọc theo 2 tầng song song:

```text
[Chọn nhóm "lua"]
       ↓
BM25 (sparse)  →  pre-filter: chỉ score docs thuộc nhóm "lua"
Pinecone (dense) → over-fetch rồi post-filter theo metadata
       ↓
RRF Fusion: gộp kết quả, rank lại
       ↓
Chỉ trả về chunk thuộc nhóm "lua"
```

> **Lưu ý kỹ thuật:** Pinecone hỗ trợ metadata filter ở dense search. Hệ thống vẫn lấy pool ứng viên đủ lớn rồi lọc theo `knowledge_group` để kết hợp ổn định với BM25.

### Tổ chức file

**Cách 1 — Upload qua frontend (khuyến nghị)**

1. Nhập tên nhóm vào ô `Nhóm tri thức khi upload`, ví dụ:

   ```text
   lua
   ca-phe
   sau-benh
   phan-bon
   ```

2. Chọn file PDF và bấm `Upload`.
3. Backend tự động lưu vào `data/uploads/<nhóm>/`.
4. Bấm `Index` sau khi upload xong.

**Cách 2 — Copy file thủ công**

Tổ chức theo cấu trúc subfolder, mỗi subfolder là một nhóm:

```text
data/uploads/
  lua/
    ky-thuat-trong-lua.pdf
    phong-tru-sau-benh-lua.pdf
  ca-phe/
    ky-thuat-ca-phe.pdf
  phan-bon/
    huong-dan-bon-phan.pdf
```

Sau đó bấm `Index` trên frontend hoặc chạy CLI.

> **Quan trọng — cấu trúc thư mục phải đúng:**
>
> | Đường dẫn | Kết quả |
> |-----------|---------|
> | `data/uploads/lua/tai-lieu.pdf` ✅ | Gắn `knowledge_group = lua` |
> | `data/uploads/tai-lieu.pdf` ⚠️ | Gắn `knowledge_group = general` (không filter được) |
> | `data/tai-lieu.pdf` ❌ | Không tìm thấy, báo lỗi |

### Trạng thái file hiện tại trong project

```text
data/uploads/
  general/
    lua_va_benh_lua_co_ban.pdf
    nong_nghiep_co_ban_RAG.pdf
```

Cả 2 file đang ở nhóm `general`. Để filter theo nhóm riêng, di chuyển thủ công rồi chạy lại `Index`:

```powershell
# Tách thành 2 nhóm riêng
mkdir data\uploads\lua
mkdir data\uploads\nong-nghiep-co-ban
Move-Item data\uploads\general\lua_va_benh_lua_co_ban.pdf data\uploads\lua\
Move-Item data\uploads\general\nong_nghiep_co_ban_RAG.pdf data\uploads\nong-nghiep-co-ban\
```

### Lỗi thường gặp

| Lỗi | Nguyên nhân | Cách sửa |
|-----|------------|----------|
| `Không tìm thấy file PDF nào` | Thư mục `data/uploads/` rỗng hoặc PDF đặt sai chỗ | Upload qua frontend hoặc copy vào `data/uploads/<nhóm>/` |
| `Thư mục chưa tồn tại` | Truyền sai đường dẫn vào ô source | Đảm bảo dùng `./data/uploads` |
| Tất cả chunk bị gắn nhóm `general` | PDF đặt thẳng trong `data/uploads/` không có subfolder | Di chuyển vào subfolder tương ứng |
| Filter nhóm không có kết quả | Nhóm quá nhỏ hoặc metadata không đồng nhất | Kiểm tra `knowledge_group` và thêm tài liệu vào nhóm đó |

### Ví dụ câu hỏi theo nhóm

```text
[Nhóm: lua]
Cách phòng bệnh đạo ôn trên lúa?
Khi nào nên bón phân cho lúa giai đoạn đẻ nhánh?

[Nhóm: ca-phe]
Dấu hiệu thiếu kali trên cây cà phê là gì?

[Tất cả]
So sánh kỹ thuật trồng lúa và cà phê?
```

## Chạy CLI

Indexing:

```powershell
python -m pipeline.indexing_pipeline --source ./data/uploads --config config.yaml
```

> **Lưu ý:** Truyền đúng `./data/uploads` (không phải `./data`). Nếu truyền folder cha, hệ thống sẽ không thể suy ra tên nhóm từ subfolder và toàn bộ file sẽ bị gắn group mặc định `general`.

Generation CLI cần một file pickle chứa `vdb_result`, nên flow React/FastAPI phù hợp hơn cho chatbot.

## Test

```powershell
pytest
```

Hoặc theo stage:

```powershell
pytest loader/tests
pytest chunking/tests
pytest embedding/tests
pytest vector_db/tests
pytest pre_retrieval/tests
pytest retrieval/tests
pytest post_retrieval/tests
pytest prompt/tests
pytest generation/tests
pytest pipeline/tests
```

## Ghi Chú Thiết Kế

- React không chứa logic RAG, chỉ gọi API.
- FastAPI gọi `IndexingPipeline` và `GenerationPipeline`.
- Pinecone lưu dense vectors; BM25 documents được giữ trên persistent storage.
- Cohere và Gemini dùng trial/free quota phù hợp cho demo; production cần theo dõi billing và rate limit.
- Không dùng OCR.
- Không dùng reranker để giữ tốc độ nhẹ.
