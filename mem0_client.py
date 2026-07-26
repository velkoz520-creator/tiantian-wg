"""
Ombre Brain (OB) 海马体记忆客户端
================================================
替代原 Mem0/Pinecone 实现。对外接口保持不变（search_mem0_context / write_mem0_chat），
内部改为通过 MCP streamable HTTP 调用 OB 的 /mcp 端点。

- 检索：search_mem0_context(query) → OB breath_search
- 写入：write_mem0_chat(user, assistant) → 后台模型判断"值得记"才调 OB hold
  （OB 哲学：只记值得记的，不因普通聊天自行写入；与 Mem0 "每轮硬塞"不同）

环境变量：
  OB_MCP_URL    OB 的 /mcp 端点（必填）
  OB_MCP_TOKEN  静态 Token（可空，空=免认证）
  BG_CHAT_BASE_URL / BG_CHAT_API_KEY / BG_BOT_MODEL  后台判断模型（建议挂 DeepSeek）
"""
import os
import json
import logging
import httpx

log = logging.getLogger("ob_client")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] ob_client: %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False

OB_MCP_URL = os.environ.get("OB_MCP_URL", "").strip()
OB_MCP_TOKEN = os.environ.get("OB_MCP_TOKEN", "").strip()

# 后台判断模型（用 BG_*，建议挂 DeepSeek 这种便宜的干杂活）
BG_BASE_URL = os.environ.get("BG_CHAT_BASE_URL", "").strip()
BG_API_KEY = os.environ.get("BG_CHAT_API_KEY", "").strip()
BG_MODEL = os.environ.get("BG_BOT_MODEL", "").strip()

_http = httpx.Client(timeout=httpx.Timeout(90.0, connect=15.0))
_session_id = None


def _headers():
    h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    if OB_MCP_TOKEN:
        h["Authorization"] = f"Bearer {OB_MCP_TOKEN}"
    if _session_id:
        h["mcp-session-id"] = _session_id
    return h


def _ensure_session():
    """首次调用时 initialize，缓存 mcp-session-id。"""
    global _session_id
    if _session_id:
        return
    body = {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "tiantian-wg", "version": "1.0"},
        },
    }
    r = _http.post(OB_MCP_URL, headers=_headers(), json=body)
    r.raise_for_status()
    sid = r.headers.get("mcp-session-id")
    if sid:
        _session_id = sid
    # initialized 通知（OB 不要求严格响应，吞掉异常）
    try:
        _http.post(OB_MCP_URL, headers=_headers(),
                   json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    except Exception:
        pass


def _parse_result(text: str) -> str:
    """从 SSE 或纯 JSON 响应里提取 tools/call 的结果文本。"""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            payload = line[5:].strip()
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            if "result" in obj:
                result = obj["result"]
                if isinstance(result, dict):
                    content = result.get("content")
                    if isinstance(content, list) and content:
                        txt = content[0].get("text")
                        if txt:
                            return txt
                    sc = result.get("structuredContent")
                    if isinstance(sc, dict) and sc.get("result"):
                        return sc["result"]
                    return json.dumps(result, ensure_ascii=False)
                return str(result)
    try:
        obj = json.loads(text)
        return json.dumps(obj.get("result", obj), ensure_ascii=False)
    except Exception:
        return text


def _call_tool(name: str, arguments: dict) -> str:
    """调 OB 的一个 MCP 工具。自动管理 session，失败重置重试一次。"""
    global _session_id
    for attempt in range(2):
        try:
            _ensure_session()
            body = {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
            r = _http.post(OB_MCP_URL, headers=_headers(), json=body)
            # session 失效 → 重置重试
            if r.status_code in (400, 404) and attempt == 0:
                _session_id = None
                continue
            r.raise_for_status()
            return _parse_result(r.text)
        except Exception as e:
            log.error("[OB] _call_tool %s 失败 attempt=%d: %s", name, attempt, e)
            _session_id = None
            if attempt == 1:
                return ""
    return ""


# ============== 对外接口（与原 mem0_client 兼容） ==============

class OBMCPClient:
    """兼容原 HybridMemoryClient 的接口，内部走 OB。"""

    def __init__(self):
        self._ok = bool(OB_MCP_URL)

    @property
    def available(self) -> bool:
        return self._ok

    def search(self, query: str, user_id: str = None, limit: int = 3) -> list:
        if not query or not query.strip():
            return []
        raw = _call_tool("breath_search", {"query": query, "max_results": limit})
        if not raw:
            return []
        # OB 返回的文本里多个桶以 "--- xxx ---" 分隔，拆成多条让注入更干净
        parts = [p.strip() for p in raw.split("---") if p.strip()]
        if len(parts) <= 1:
            return [{"memory": raw}]
        return [{"memory": p} for p in parts]

    def add(self, messages: list, user_id: str = None) -> bool:
        user_text = ""
        assistant_text = ""
        for m in messages:
            if not isinstance(m, dict):
                continue
            if m.get("role") == "user":
                user_text = str(m.get("content", ""))
            elif m.get("role") == "assistant":
                assistant_text = str(m.get("content", ""))
        if not user_text:
            return False
        return _write_with_judgment(user_text, assistant_text)


def search_mem0_context(query: str, limit: int = 3) -> str:
    """用 query 检索 OB，返回可直接注入 context 的文本。"""
    items = mem0.search(query, limit=limit)
    if not items:
        return ""
    lines = []
    for m in items:
        text = m.get("memory", str(m)) if isinstance(m, dict) else str(m)
        lines.append(f"- {text}")
    return "【深层记忆（OB 海马体检索）】\n" + "\n".join(lines)


def _judge_worth_remembering(user_text: str, assistant_text: str) -> str:
    """用后台模型判断这轮对话值不值得记。
    返回：值得记 → 一句话客观事实；不值得 → 空串。"""
    if not (BG_BASE_URL and BG_API_KEY and BG_MODEL):
        # 后台模型没配：保守不记，避免污染记忆库
        log.warning("[OB] 后台判断模型未配置(BG_CHAT_*)，跳过判断，本轮不写入")
        return ""
    prompt = (
        "判断下面这轮对话有没有【值得长期记住】的内容（事实/偏好/重要事件/关系进展/承诺等）。"
        "日常寒暄、废话、纯情绪发泄不值得记。\n"
        "- 值得记：提取成一句客观事实（保留原意、不要摘要腔），如「天天喜欢X」「天天和X约定了Y」。"
        "用第三人称，主语用对方的名字。\n"
        "- 不值得记：只回复 {\"remember\": false}\n"
        f"user: {user_text[:500]}\nassistant: {assistant_text[:500]}\n"
        "只输出 JSON：{\"remember\": true, \"content\": \"一句话\"} 或 {\"remember\": false}"
    )
    try:
        r = _http.post(
            f"{BG_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {BG_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": BG_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 200,
            },
            timeout=30,
        )
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"].strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start < 0 or end <= 0:
            return ""
        obj = json.loads(raw[start:end])
        if obj.get("remember") and obj.get("content"):
            return str(obj["content"]).strip()
        return ""
    except Exception as e:
        log.error("[OB] 判断 LLM 调用失败: %s", e)
        return ""


def _write_with_judgment(user_text: str, assistant_text: str) -> bool:
    """判断值得记才调 OB hold 写入。"""
    if not user_text:
        return False
    content = _judge_worth_remembering(user_text, assistant_text)
    if not content:
        log.info("[OB] 本轮判断为不值得记，跳过写入 user=%r", user_text[:60])
        return False
    result = _call_tool("hold", {"content": content, "importance": 5})
    ok = bool(result) and ("error" not in result.lower()[:50])
    if ok:
        log.info("[OB] 写入成功 content=%r", content[:80])
    else:
        log.error("[OB] 写入失败 result=%s", result[:200])
    return ok


def write_mem0_chat(user_text: str, assistant_text: str):
    """把一轮对话写入 OB 形成长期记忆（带判断）。"""
    if not mem0.available:
        log.warning("[write_mem0_chat] OB 不可用(OB_MCP_URL 未配)，跳过")
        return
    if not user_text:
        log.warning("[write_mem0_chat] user_text 为空，跳过")
        return
    try:
        _write_with_judgment(user_text, assistant_text)
    except Exception as e:
        log.error("[write_mem0_chat] 异常: %s", e)


# 全局单例（保持与原模块兼容，workers.py 等处 import mem0）
mem0 = OBMCPClient()
