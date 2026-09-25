import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { Bot, Database, FileUp, Leaf, Loader2, Send, Server, Upload } from "lucide-react";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || "Request failed");
  }
  return data;
}

function App() {
  const [status, setStatus] = useState({ ready: false, groups: [] });
  const [sourcePath, setSourcePath] = useState("./data/uploads");
  const [uploadGroup, setUploadGroup] = useState("general");
  const [uploadLocation, setUploadLocation] = useState("");
  const [selectedGroup, setSelectedGroup] = useState("all");
  const [files, setFiles] = useState([]);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");

  const readyText = useMemo(() => {
    if (!status.ready) return "Chưa có index";
    return `${status.n_docs || 0} docs · ${status.n_chunks || 0} chunks · ${status.n_vectors || 0} vectors`;
  }, [status]);

  const groupOptions = useMemo(() => {
    const groups = new Set(status.groups || []);
    if (selectedGroup !== "all") groups.add(selectedGroup);
    return Array.from(groups).sort();
  }, [status.groups, selectedGroup]);

  async function refreshStatus() {
    try {
      const data = await api("/indexing/status");
      // Route through applyIndexStatus so selectedGroup is kept in sync
      // with the groups that actually exist in the current index.
      applyIndexStatus({ success: data.ready, ...data });
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => {
    refreshStatus();
  }, []);

  async function uploadFiles() {
    if (!files.length) return;
    setBusy("upload");
    setError("");
    try {
      const form = new FormData();
      Array.from(files).forEach((file) => form.append("files", file));
      form.append("group", uploadGroup || "general");
      const data = await api("/indexing/upload", { method: "POST", body: form });
      setSourcePath(data.source_path || "./data/uploads");
      setUploadLocation(data.group_path || "");
      setSelectedGroup(data.group || "all");
      await refreshStatus();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy("");
    }
  }

  function applyIndexStatus(data) {
    setStatus({ ready: data.success, ...data });
    if (data.groups?.length && selectedGroup !== "all" && !data.groups.includes(selectedGroup)) {
      setSelectedGroup("all");
    }
  }

  async function runIndexing() {
    setBusy("indexing");
    setError("");
    try {
      const data = await api("/indexing/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_path: sourcePath }),
      });
      applyIndexStatus(data);
      if (!data.success) setError(data.error || "Indexing failed");
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy("");
    }
  }

  async function loadExistingIndex() {
    setBusy("load");
    setError("");
    try {
      const data = await api("/indexing/load", { method: "POST" });
      applyIndexStatus({ loaded_from_disk: true, ...data });
      if (!data.success) setError(data.error || "Load index failed");
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy("");
    }
  }

  async function askQuestion(event) {
    event.preventDefault();
    const text = question.trim();
    if (!text) return;

    setQuestion("");
    setError("");
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setBusy("chat");

    try {
      // Build history from current messages so the backend can use
      // conversation context for better answers.
      const history = messages
        .filter((m) => m.role === "user" || m.role === "assistant")
        .map((m) => ({ role: m.role, content: m.content }));

      const data = await api("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: text,
          knowledge_group: selectedGroup === "all" ? null : selectedGroup,
          history,
        }),
      });
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer || "Đã lấy được ngữ cảnh, nhưng chưa có câu trả lời từ model.",
          sources: data.sources || [],
        },
      ]);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy("");
    }
  }

  return (
    <main className="app-shell">
      <aside className="side-panel">
        <div className="brand">
          <span className="brand-mark"><Leaf size={20} /></span>
          <div>
            <h1>RAG Nông Nghiệp</h1>
            <p>PDF · Cohere · Pinecone · Gemini Flash</p>
          </div>
        </div>

        <section className="panel-section">
          <div className="section-title"><Database size={17} /> Indexing</div>
          <label className="field-label" htmlFor="source">PDF file hoặc folder</label>
          <input id="source" value={sourcePath} onChange={(e) => setSourcePath(e.target.value)} />

          <label className="field-label" htmlFor="group">Nhóm tri thức khi upload</label>
          <input
            id="group"
            value={uploadGroup}
            onChange={(e) => setUploadGroup(e.target.value)}
            placeholder="general, lua, ca-phe, sau-benh..."
          />

          <label className="upload-box">
            <FileUp size={20} />
            <span>{files.length ? `${files.length} PDF đã chọn` : "Chọn PDF"}</span>
            <input type="file" accept="application/pdf" multiple onChange={(e) => setFiles(e.target.files)} />
          </label>
          {uploadLocation && <div className="hint">Đã upload vào: {uploadLocation}</div>}

          <div className="button-row three">
            <button type="button" onClick={uploadFiles} disabled={!files.length || busy}>
              {busy === "upload" ? <Loader2 className="spin" size={16} /> : <Upload size={16} />}
              Upload
            </button>
            <button type="button" onClick={loadExistingIndex} disabled={busy}>
              {busy === "load" ? <Loader2 className="spin" size={16} /> : <Server size={16} />}
              Load
            </button>
            <button type="button" className="primary" onClick={runIndexing} disabled={busy}>
              {busy === "indexing" ? <Loader2 className="spin" size={16} /> : <Database size={16} />}
              Index
            </button>
          </div>
        </section>

        <section className="panel-section compact">
          <div className="section-title"><Server size={17} /> Trạng thái</div>
          <div className={status.ready ? "status ready" : "status"}>{readyText}</div>
          <label className="field-label" htmlFor="search-group">Tìm trong nhóm</label>
          <select
            id="search-group"
            value={selectedGroup}
            onChange={(e) => setSelectedGroup(e.target.value)}
          >
            <option value="all">Tất cả</option>
            {groupOptions.map((group) => (
              <option key={group} value={group}>{group}</option>
            ))}
          </select>
          <button type="button" onClick={refreshStatus} disabled={busy}>Refresh</button>
        </section>
      </aside>

      <section className="chat-panel">
        <header className="chat-header">
          <div>
            <h2>Chatbot tư vấn tài liệu nông nghiệp</h2>
            <p>Trả lời dựa trên PDF đã index và kèm nguồn tham chiếu.</p>
          </div>
          <span className={status.ready ? "pill ok" : "pill"}>{status.ready ? "Ready" : "No index"}</span>
        </header>

        {error && <div className="error">{error}</div>}

        <div className="messages">
          {!messages.length && (
            <div className="empty-state">
              <Bot size={34} />
              <p>Nhập câu hỏi sau khi index tài liệu PDF.</p>
            </div>
          )}

          {messages.map((message, index) => (
            <article className={`message ${message.role}`} key={`${message.role}-${index}`}>
              <div className="bubble">{message.content}</div>
              {!!message.sources?.length && (
                <div className="sources">
                  <div className="sources-title">Nguồn tham khảo ({message.sources.length})</div>
                  {message.sources.map((source, sourceIndex) => (
                    <details key={`${source.file_name}-${sourceIndex}`}>
                      <summary>
                        [{sourceIndex + 1}] {source.file_name || source.source || "PDF"}
                        {source.page_number ? ` · trang ${source.page_number}` : ""}
                        {source.knowledge_group ? ` · ${source.knowledge_group}` : ""}
                        {source.title || source.section ? ` · ${source.title || source.section}` : ""}
                      </summary>
                      <p>{source.text}</p>
                    </details>
                  ))}
                </div>
              )}
            </article>
          ))}
        </div>

        <form className="composer" onSubmit={askQuestion}>
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ví dụ: Cách phòng bệnh đạo ôn trên lúa?"
            disabled={busy === "chat"}
          />
          <button type="submit" className="primary icon-button" disabled={busy === "chat" || !question.trim()}>
            {busy === "chat" ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
          </button>
        </form>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
