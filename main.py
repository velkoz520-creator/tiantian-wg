import os
import json
import time
import hmac
import hashlib
import asyncio
import httpx
import logging
import requests
import threading
import warnings
import uvicorn
from datetime import timezone, timedelta
from urllib.parse import parse_qs
from starlette.applications import Starlette
from starlette.requests import ClientDisconnect
from starlette.types import ASGIApp, Scope, Receive, Send
 
# 全局日志配置：必须在其他本地模块 import 之前调用，确保所有模块用
# logging.getLogger(__name__) 拿到的 logger（wx_bot/context/scheduled 等）
# 在没有额外配置的情况下也能把 INFO 级别以上的日志输出到 stdout。
#
# 背景：之前项目里从未调用过 logging.basicConfig()。Python logging 在没有
# 配置 handler 时，只会触发内置的 lastResort handler（级别 WARNING），
# 也就是说所有 log.info(...) 调用此前实际上被静默丢弃、根本不会出现在
# Zeabur 日志里——即使代码里"看起来"打了日志。这里显式配置一次，
# force=True 确保覆盖掉任何第三方库可能提前设置的 handler。
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
 
from utils import get_active_llm_config
from context import build_rikkahub_context, save_chat_message, save_rikkahub_message
from mem0_client import write_mem0_chat, search_mem0_context
from bg_executor import submit_background, track_task
from prompts import AI_NAME, PARTNER_NAME, RIKKAHUB_PLATFORM_AWARENESS
 
# 以下几条 DeprecationWarning 都来自第三方库内部实现（supabase-py 创建 httpx
# client 时用了旧参数、uvicorn 还在用旧版 websockets API），不是本项目代码
# 直接触发的，也不影响服务运行（不修就是纯日志噪音）。这里只精确过滤这三个
# 来源模块，不会连带盖住本项目代码或其他库以后可能产生的 DeprecationWarning。
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"supabase.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"websockets.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"uvicorn.*")
 
API_SECRET   = os.environ.get("API_SECRET", "")
GATEWAY_HOST = os.environ.get("GATEWAY_HOST", "")
 
# 上游 LLM 流式响应的读超时（秒）：指"两次收到数据之间的最大间隔"，不是总时长。
# 之前是 read=None 完全无限等待，上游排队/假死时客户端只能永远转圈。
# 推理类模型思考期通常也会持续吐 reasoning delta 或 keepalive，180 秒完全
# 收不到任何字节基本可以断定这次请求已经废了，断开并提示重试比干等强。
UPSTREAM_READ_TIMEOUT = int(os.environ.get("UPSTREAM_READ_TIMEOUT", "180"))
 
INIT_DATA_MAX_AGE = 86400
 
log = logging.getLogger(__name__)
 
 
def _load_miniapp_html():
    path = os.path.join(os.path.dirname(__file__), "miniapp.html")
    with open(path, "rb") as f:
        return f.read()
 
 
def _extract_last_user_message(messages: list) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part.get("text", "")
            elif isinstance(content, str):
                return content
    return ""
 
 
def _verify_telegram_init_data(init_data: str, bot_token: str) -> dict | None:
    if not init_data or not bot_token:
        return None
    try:
        parsed = parse_qs(init_data)
        recv_hash_list = parsed.pop("hash", None)
        if not recv_hash_list:
            return None
        recv_hash = recv_hash_list[0]
 
        auth_date_list = parsed.get("auth_date")
        if not auth_date_list:
            return None
        auth_date = int(auth_date_list[0])
        if time.time() - auth_date > INIT_DATA_MAX_AGE:
            return None
 
        sorted_items = sorted((k, v[0]) for k, v in parsed.items())
        data_check_string = "\n".join(f"{k}={v}" for k, v in sorted_items)
 
        secret = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
        computed = hmac.new(secret, data_check_string.encode(), hashlib.sha256).hexdigest()
 
        if not hmac.compare_digest(computed, recv_hash):
            return None
 
        user_list = parsed.get("user")
        return json.loads(user_list[0]) if user_list else None
    except Exception as e:
        log.error("[Gateway] initData 校验异常: %s", e, exc_info=True)
        return None
 
 
async def _read_body(receive) -> bytes:
    """
    按 ASGI 协议读取完整请求体。
 
    关键点：receive() 在客户端中途断开连接时会返回
    {"type": "http.disconnect"}（没有 body / more_body 字段），
    必须显式识别这种消息，否则会把"还没收完的半截数据"误判为
    "已经收完的完整数据"，传给上层做 JSON 解析时只会得到一个
    UnicodeDecodeError/JSONDecodeError，而那其实是连接中断，不是
    请求格式问题。
    """
    body = b""
    while True:
        msg = await receive()
        if msg.get("type") == "http.disconnect":
            raise ClientDisconnect()
        body += msg.get("body", b"")
        if not msg.get("more_body", False):
            break
    return body
 
 
async def _safe_send(send, message: dict, context: str = "") -> bool:
    """
    发送一条 ASGI 消息。如果客户端已经断开连接，send() 会抛出
    ClientDisconnect 或服务器特定的 OSError 子类（uvicorn 是
    ClientDisconnected）。这里统一吞掉这一类"预期内"的异常，
    记录一条 warning 后返回 False，而不是让异常继续往外抛——
    那样只会变成一条无意义的 500 / unhandled exception 日志噪音，
    对一个已经不在线的客户端毫无意义。
 
    返回 True 表示发送成功；False 表示客户端已经不在线，调用方
    应该停止后续的发送（没有意义），但仍然可以继续做与客户端无关
    的收尾工作（比如把已经生成的内容存历史）。
    """
    try:
        await send(message)
        return True
    except (ClientDisconnect, OSError) as e:
        log.warning("[Gateway] 发送响应时客户端已断开连接 context=%s: %s", context or "?", e)
        return False
 
 
async def _send_json(send, status: int, payload: dict):
    body = json.dumps(payload).encode("utf-8")
    if not await _safe_send(send, {
        "type": "http.response.start", "status": status,
        "headers": [(b"content-type", b"application/json")],
    }, context=f"_send_json status={status}"):
        return
    await _safe_send(send, {"type": "http.response.body", "body": body},
                      context=f"_send_json status={status}")
 
 
class RikkahubGatewayMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app
 
    async def __call__(self, scope: Scope, receive, send):
        path = scope.get("path", "")
 
        if scope["type"] == "websocket" and path == "/qq-ws":
            from qq_bot import handle_napcat_ws_forward
            await handle_napcat_ws_forward(scope, receive, send)
            return
 
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
 
        path = scope["path"]
 
        if path == "/miniapp":
            html = await asyncio.to_thread(_load_miniapp_html)
            await _safe_send(send, {"type": "http.response.start", "status": 200,
                        "headers": [(b"content-type", b"text/html; charset=utf-8")]},
                        context="/miniapp")
            await _safe_send(send, {"type": "http.response.body", "body": html}, context="/miniapp")
            return
 
        if path == "/auth":
            if scope["method"] != "POST":
                await _send_json(send, 405, {"error": "method not allowed"})
                return
 
            try:
                body = await _read_body(receive)
            except ClientDisconnect:
                log.warning("[Gateway] /auth 客户端在发送请求体过程中断开连接，终止处理")
                return
 
            try:
                req = json.loads(body.decode("utf-8")) if body else {}
                init_data = req.get("initData", "")
 
                bot_token    = os.environ.get("TG_BOT_TOKEN", "")
                tg_chat_id   = os.environ.get("TG_CHAT_ID", "")
                supabase_url = os.environ.get("SUPABASE_URL", "")
                supabase_key = os.environ.get("SUPABASE_KEY", "")
 
                user = _verify_telegram_init_data(init_data, bot_token)
                if not user:
                    await _send_json(send, 401, {"error": "invalid initData"})
                    return
                if str(user.get("id", "")) != str(tg_chat_id):
                    await _send_json(send, 403, {"error": "user not allowed"})
                    return
 
                await _send_json(send, 200, {
                    "supabase_url": supabase_url,
                    "supabase_key": supabase_key,
                    "ai_name": AI_NAME,
                    "partner_name": PARTNER_NAME,
                })
                return
            except (UnicodeDecodeError, json.JSONDecodeError) as parse_err:
                log.warning(
                    "[Gateway] /auth 请求体不是合法 JSON: %s | body_len=%d | body_preview=%r",
                    parse_err, len(body), body[:200], exc_info=True,
                )
                await _send_json(send, 400, {"error": f"invalid JSON body: {parse_err}"})
                return
            except Exception as e:
                log.error("[Gateway] /auth 处理异常: %s", e, exc_info=True)
                await _send_json(send, 500, {"error": "internal"})
                return
 
        if path == "/webhook":
            if scope["method"] != "POST":
                await _send_json(send, 405, {"error": "method not allowed"})
                return
 
            try:
                body = await _read_body(receive)
            except ClientDisconnect:
                log.warning("[Gateway] /webhook 客户端在发送请求体过程中断开连接，终止处理")
                return
 
            try:
                update = json.loads(body.decode("utf-8")) if body else {}
            except (UnicodeDecodeError, json.JSONDecodeError) as parse_err:
                log.warning(
                    "[Gateway] /webhook 请求体不是合法 JSON: %s | body_len=%d | body_preview=%r",
                    parse_err, len(body), body[:200], exc_info=True,
                )
                await _send_json(send, 400, {"error": "bad request"})
                return
 
            # 验证来源（用 API_SECRET 作为 secret_token）
            headers_dict = {
                k.decode("utf-8").lower(): v.decode("utf-8")
                for k, v in scope.get("headers", [])
            }
            incoming_secret = headers_dict.get("x-telegram-bot-api-secret-token", "")
            api_secret = API_SECRET
            if api_secret:
                import re as _re
                clean_secret = _re.sub(r'[^A-Za-z0-9_\-]', '', api_secret)[:256]
                if clean_secret and incoming_secret != clean_secret:
                    log.warning(
                        "[Gateway] /webhook secret token 不匹配，已拒绝 incoming=%s...",
                        incoming_secret[:10] if incoming_secret else "(空)",
                    )
                    await _send_json(send, 403, {"error": "forbidden"})
                    return
 
            # 立即返回 200，异步处理
            await _send_json(send, 200, {"ok": True})
 
            if update:
                from workers import handle_telegram_update
                track_task(asyncio.create_task(handle_telegram_update(update)))
            else:
                log.warning("[Gateway] /webhook update 为空，跳过处理")
            return
 
        if path.endswith("/v1/chat/completions"):
 
            if scope["method"] == "OPTIONS":
                if await _safe_send(send, {"type": "http.response.start", "status": 200, "headers": [
                    (b"access-control-allow-origin", b"*"),
                    (b"access-control-allow-methods", b"POST, OPTIONS"),
                    (b"access-control-allow-headers", b"content-type, authorization"),
                ]}, context="OPTIONS preflight"):
                    await _safe_send(send, {"type": "http.response.body", "body": b""},
                                      context="OPTIONS preflight body")
                return
 
            if scope["method"] == "POST":
                headers_dict = {
                    k.decode("utf-8").lower(): v.decode("utf-8")
                    for k, v in scope.get("headers", [])
                }
                auth_header = headers_dict.get("authorization", "")
                auth_token = auth_header.split(" ", 1)[-1].strip() \
                    if " " in auth_header else auth_header.strip()
 
                if API_SECRET and auth_token != API_SECRET:
                    await _send_json(send, 401, {"error": {"message": "Unauthorized"}})
                    return
 
                try:
                    body = await _read_body(receive)
                except ClientDisconnect:
                    log.warning(
                        "[Gateway] /v1/chat/completions 客户端在发送请求体过程中断开连接"
                        "（请求体未完整接收，不是 JSON 格式问题），已终止处理，不再尝试发送响应"
                    )
                    return
 
                if not body:
                    log.warning("[Gateway] /v1/chat/completions 收到空请求体，已拒绝处理")
                    await _send_json(send, 400, {
                        "error": {"message": "request body is empty", "code": 400}
                    })
                    return
 
                try:
                    req_data = json.loads(body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as parse_err:
                    log.warning(
                        "[Gateway] /v1/chat/completions 请求体不是合法 JSON: %s | body_len=%d | body_preview=%r",
                        parse_err, len(body), body[:200], exc_info=True,
                    )
                    await _send_json(send, 400, {
                        "error": {"message": f"invalid JSON body: {parse_err}", "code": 400}
                    })
                    return
 
                try:
                    original_messages = req_data.get("messages", [])
 
                    user_text = _extract_last_user_message(original_messages)
                    # rikkahub 每完成一轮工具调用都会带着 role=tool 的结果重新发起
                    # 一次全新请求，此时"最后一条 user 消息"仍是同一句话。
                    # 记录消息列表的尾部角色：只有以 user 结尾（真正的新用户输入）
                    # 的请求才允许把 user 消息写入历史，否则同一句话会随工具轮数
                    # 被重复入库，历史和 Mem0 越滚越大、context 越来越慢。
                    last_msg_role = original_messages[-1].get("role", "") if original_messages else ""
 
                    # 先检测是否主动消息（决定用精简版还是完整版 context）
                    _is_proactive = False
                    if original_messages and original_messages[0].get("role") == "system":
                        existing_system = original_messages[0].get("content", "")
                        _is_proactive = "[主动消息上下文]" in existing_system

                    # 主动消息用 lean=True（精简注入，避免独白体淹没主动指令）
                    system_prompt = await asyncio.to_thread(build_rikkahub_context, True, _is_proactive)
                    # v2 第一批第1条：橘瓣端平台与能力感知（仅橘瓣，TG 走 build_bot_context 不经过这里）
                    system_prompt = RIKKAHUB_PLATFORM_AWARENESS + "\n\n" + system_prompt

                    # v2 痛点A修复：橘瓣端 OB 记忆检索（之前橘瓣只写不读，跨平台断片）
                    # 照抄 TG 端 workers.py:335 的写法，用最后一条 user 消息当 query 调 OB breath_search。
                    # 这样橘瓣克老师能读到 TG/其他端写进 OB 的记忆，实现跨平台记忆连续。
                    if user_text:
                        mem0_ctx = await asyncio.to_thread(search_mem0_context, user_text, 3)
                        if mem0_ctx:
                            system_prompt += "\n\n" + mem0_ctx

                    if original_messages and original_messages[0].get("role") == "system":
                        existing_system = original_messages[0].get("content", "")
                        if _is_proactive:
                            # 主动消息：网关人格在前，橘瓣主动指令在后（A方案，让克老师最后读到主动任务）
                            original_messages[0]["content"] = system_prompt + "\n\n" + existing_system
                            log.info("[Gateway] 检测到橘瓣主动消息，精简注入+主动指令置后")
                        else:
                            original_messages[0]["content"] = system_prompt
                    else:
                        original_messages.insert(0, {
                            "role": "system", "content": system_prompt})
                    req_data["messages"] = original_messages
 
                    cfg = await asyncio.to_thread(get_active_llm_config)
                    target_url = cfg["base_url"] + "/chat/completions"
                    upstream_headers = {
                        "Authorization": f"Bearer {cfg['api_key']}",
                        "Content-Type":  "application/json",
                        **cfg.get("extra_headers", {}),
                    }
 
                    req_data["stream"] = True
                    req_data.pop("stream_options", None)
                    req_data["model"] = cfg["model"]
 
                    if not await _safe_send(send, {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [
                            (b"content-type", b"text/event-stream; charset=utf-8"),
                            (b"cache-control", b"no-cache"),
                            (b"connection", b"keep-alive"),
                            (b"access-control-allow-origin", b"*"),
                        ],
                    }, context="chat/completions response.start"):
                        # 客户端在我们准备好响应头之前就已经断开了，
                        # 没必要再去调用上游 LLM 浪费一次请求额度
                        return
 
                    assistant_buf = []
                    client_gone = False
                    has_tool_calls = False
 
                    try:
                        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=30, read=UPSTREAM_READ_TIMEOUT, write=30, pool=30), http2=False) as client:
                            async with client.stream(
                                "POST", target_url,
                                headers=upstream_headers,
                                json=req_data,
                            ) as resp:
                                if resp.status_code >= 400:
                                    err_body = await resp.aread()
                                    body_text = err_body.decode("utf-8", errors="ignore")[:300]
                                    # 诊断：400时打出请求的关键结构，定位是模型/图片/格式哪个出问题
                                    _diag_msgs = req_data.get("messages", [])
                                    _diag_has_image = any(
                                        isinstance(m.get("content"), list) and
                                        any(isinstance(p, dict) and p.get("type") == "image_url" for p in m["content"])
                                        for m in _diag_msgs
                                    )
                                    _diag_msg_roles = [m.get("role") for m in _diag_msgs]
                                    _diag_model = req_data.get("model", "(空)")
                                    _diag_msg_count = len(_diag_msgs)
                                    # 如果有图片，统计图片大小（不打base64全文，只打长度）
                                    _diag_img_info = ""
                                    for m in _diag_msgs:
                                        if isinstance(m.get("content"), list):
                                            for p in m["content"]:
                                                if isinstance(p, dict) and p.get("type") == "image_url":
                                                    url = p.get("image_url", {}).get("url", "")
                                                    _diag_img_info = f"图片url长度={len(url)} 前缀={url[:30]}"
                                    log.error(
                                        "[Gateway] 上游错误 status=%s model=%s | 诊断: msg数=%s roles=%s 含图片=%s %s | 上游body=%s",
                                        resp.status_code, _diag_model, _diag_msg_count, _diag_msg_roles,
                                        _diag_has_image, _diag_img_info, body_text,
                                    )
                                    err_msg = "请求过于频繁，请稍后重试 (429)" if resp.status_code == 429 else f"上游服务错误 ({resp.status_code})"
                                    err_chunk = json.dumps({
                                        "choices": [{
                                            "index": 0,
                                            "delta": {"content": f"\n\n[{err_msg}]"},
                                            "finish_reason": "stop",
                                        }]
                                    })
                                    if await _safe_send(send, {
                                        "type": "http.response.body",
                                        "body": f"data: {err_chunk}\n\ndata: [DONE]\n\n".encode(),
                                        "more_body": True,
                                    }, context="upstream error chunk"):
                                        # 上面那次 send 标了 more_body=True，按 ASGI 协议必须
                                        # 再发一次 more_body=False（或省略该字段）才算正式收尾，
                                        # 否则会报 "ASGI callable returned without completing response"
                                        await _safe_send(send, {"type": "http.response.body", "body": b""},
                                                          context="upstream error chunk finalize")
                                    return
 
                                _got_done = False
                                _finish_reason = None
                                _content_chars = 0
                                _chunk_count = 0
 
                                async for line in resp.aiter_lines():
                                    if not line:
                                        continue
                                    if not line.startswith("data:"):
                                        continue
                                    data_str = line[5:].strip()
                                    if data_str == "[DONE]":
                                        _got_done = True
                                        if not await _safe_send(send, {
                                            "type": "http.response.body",
                                            "body": b"data: [DONE]\n\n",
                                            "more_body": True,
                                        }, context="DONE chunk"):
                                            client_gone = True
                                        break
                                    try:
                                        chunk = json.loads(data_str)
                                        _chunk_count += 1
                                        choice = chunk.get("choices", [{}])[0]
                                        delta = choice.get("delta", {})
                                        fr = choice.get("finish_reason")
                                        if fr:
                                            _finish_reason = fr
                                        if delta.get("content"):
                                            assistant_buf.append(delta["content"])
                                            _content_chars += len(delta["content"])
                                        if delta.get("tool_calls"):
                                            has_tool_calls = True
                                    except Exception as chunk_err:
                                        log.warning(
                                            "[Gateway] 解析上游流式 chunk 失败（仍会原样转发给客户端）: %s | data=%r",
                                            chunk_err, data_str[:200],
                                        )
                                    if not await _safe_send(send, {
                                        "type": "http.response.body",
                                        "body": (line + "\n\n").encode("utf-8"),
                                        "more_body": True,
                                    }, context="forward chunk"):
                                        client_gone = True
                                        break
 
                                if client_gone:
                                    log.info(
                                        "[Gateway] 客户端中途断开连接，停止转发上游流 chunks=%d finish_reason=%s content=%d字符",
                                        _chunk_count, _finish_reason, _content_chars,
                                    )
                                elif _got_done:
                                    log.info(
                                        "[Gateway] 流正常结束 chunks=%d finish_reason=%s content=%d字符",
                                        _chunk_count, _finish_reason, _content_chars,
                                    )
                                else:
                                    log.warning(
                                        "[Gateway] 流提前结束！未收到[DONE] chunks=%d finish_reason=%s content=%d字符",
                                        _chunk_count, _finish_reason, _content_chars,
                                    )
 
                    except httpx.ReadTimeout as stream_err:
                        log.error(
                            "[Gateway] 上游超过 %s 秒未返回任何数据，判定超时断开 target_url=%s: %s",
                            UPSTREAM_READ_TIMEOUT, target_url, stream_err, exc_info=True,
                        )
                        err_chunk = json.dumps({
                            "choices": [{
                                "index": 0,
                                "delta": {"content": f"\n\n[上游模型 {UPSTREAM_READ_TIMEOUT} 秒没有返回任何数据，已超时断开，请重试]"},
                                "finish_reason": "stop",
                            }]
                        })
                        await _safe_send(send, {
                            "type": "http.response.body",
                            "body": f"data: {err_chunk}\n\ndata: [DONE]\n\n".encode(),
                            "more_body": True,
                        }, context="read_timeout notify")
                    except Exception as stream_err:
                        log.error(
                            "[Gateway] 流式转发异常 target_url=%s: %s",
                            target_url, stream_err, exc_info=True,
                        )
                        err_chunk = json.dumps({
                            "choices": [{
                                "index": 0,
                                "delta": {"content": "\n\n[连接中断，请重试]"},
                                "finish_reason": "stop",
                            }]
                        })
                        await _safe_send(send, {
                            "type": "http.response.body",
                            "body": f"data: {err_chunk}\n\ndata: [DONE]\n\n".encode(),
                            "more_body": True,
                        }, context="stream_err notify")
 
                    await _safe_send(send, {"type": "http.response.body", "body": b""},
                                      context="chat/completions final close")
 
                    # ── 收尾持久化：全部走 bg_executor 共享线程池 ──
                    # 这里原来是逐条 threading.Thread(...).start()，且完全没有
                    # try/except 保护。此时响应已经发送过 http.response.start
                    # （200, event-stream），一旦这里抛异常会被外层 except 兜底、
                    # 尝试对同一连接二次发送 response.start，触发 ASGI 协议冲突，
                    # 导致这次响应既不能正常关闭也不会报错——客户端只能一直等数据。
                    # submit_background 内部保证不会向上抛异常（提交失败/任务
                    # 内部异常都只记日志），所以这里不需要也不应该再包一层
                    # try/except：submit_background 本身就是这段收尾逻辑的
                    # try/except 保护层。
                    assistant_text = "".join(assistant_buf).strip()
                    _stripped = user_text.strip()
                    _is_system = (
                        _stripped.startswith(f"你是{AI_NAME}")
                        or _stripped.startswith("[主动消息上下文]")
                        or _is_proactive  # 主动消息回复不存 chat_messages，防复读死循环
                    )
                    # user 消息只在"消息列表以 user 结尾"（真正的新用户输入）的那次
                    # 请求存一次；工具往返的后续请求（以 role=tool 结尾）里的
                    # "最后一条 user 消息"仍是同一句话，不再重复入库。
                    if user_text and last_msg_role == "user":
                        if _is_system:
                            submit_background(save_rikkahub_message, "user", user_text)
                        else:
                            submit_background(save_chat_message, "user", f"[rikkahub] {user_text}")
                    # assistant 每轮真实说出的话都存：工具中间轮伴随 tool_calls 的
                    # 插话（比如"让我看看！"）客户端同样展示了，是真实对话的一部分。
                    if assistant_text:
                        if _is_system:
                            submit_background(save_rikkahub_message, "assistant", assistant_text)
                        else:
                            submit_background(save_chat_message, "assistant", assistant_text)
 
                    # Mem0 只在"本轮响应没有再发起工具调用"（即这轮就是最终回复）时
                    # 写一次，避免同一组 user+assistant 随工具轮数重复写进长期记忆。
                    if user_text and assistant_text and not _is_system and not has_tool_calls:
                        submit_background(write_mem0_chat, user_text, assistant_text)
 
                    return
 
                except Exception as e:
                    log.error(
                        "[Gateway] /v1/chat/completions 处理异常: %s",
                        e, exc_info=True,
                    )
                    await _send_json(send, 500, {"error": {"message": str(e), "code": 500}})
                    return
 
        await self.app(scope, receive, send)
 
 
_bg_loop: asyncio.AbstractEventLoop | None = None
_bg_tasks: list = []
_bg_ready = threading.Event()
 
 
def start_background_tasks():
    """
    Process A（消息进程）自己的常驻协程：只保留跟 QQ/TG/微信实时消息收发直接
    相关的部分。

    2026-07-14 消息进程/后台进程拆分记录：主动思考（TG+微信）、提醒检查、
    凌晨总结、自由活动、全平台压缩+摘要整理这些自主/周期性任务，已经搬去
    background_main.py 独立进程跑（同一个容器内，entrypoint.sh 同时拉起
    main.py 和 background_main.py 两个独立操作系统进程）。这里不再启动，
    避免两边重复执行、互相抢占资源、甚至重复触发同一件事（比如两边都跑
    一次凌晨总结）。
    """
    from workers import async_telegram_polling
    from qq_bot import async_qq_bot
    from wx_bot import async_wx_bot

    global _bg_loop

    def _run():
        global _bg_loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _bg_loop = loop

        coros = [
            async_telegram_polling(),
            async_qq_bot(),
            async_wx_bot(),
        ]
        tasks = [loop.create_task(c) for c in coros]
        _bg_tasks.extend(tasks)
        _bg_ready.set()
        try:
            loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        finally:
            loop.close()

    threading.Thread(target=_run, daemon=True, name="bg-tasks-loop").start()
    _bg_ready.wait(timeout=10)
 
 
async def shutdown_background_tasks():
    """
    Starlette lifespan shutdown 钩子：uvicorn 收到部署重启/停止信号、走标准
    ASGI 生命周期关闭流程时会自动调用这里（在下面 base_app = Starlette(...,
    on_shutdown=[shutdown_background_tasks]) 里注册）。
 
    背景：start_background_tasks() 里那一批常驻后台协程（TG/QQ/微信轮询、
    主动思考、提醒检查、凌晨总结、自由活动）跑在一个独立线程自己起的独立
    事件循环里，跟 uvicorn 主服务的事件循环完全是两套体系、互不知道对方的
    存在。之前进程关闭时，uvicorn 那边走完了自己的关闭流程，但这个独立
    循环完全没收到通知，会在 Python 解释器已经开始收尾、默认线程池已关闭
    的时候，还想调度下一次任务，报 RuntimeError:
    cannot schedule new futures after shutdown。
 
    这里主动把 _bg_tasks 里所有任务 cancel 掉，并通过
    asyncio.run_coroutine_threadsafe 把"取消 + 等待收尾"这个过程调度到
    它们真正所在的那个独立事件循环去执行，在这边（uvicorn 的主事件循环）
    await 等待完成，相当于给这批后台任务补上了一个真正的关机开关。
    """
    if _bg_loop is None or not _bg_tasks:
        return
 
    async def _cancel_all():
        for t in _bg_tasks:
            t.cancel()
        await asyncio.gather(*_bg_tasks, return_exceptions=True)
 
    try:
        future = asyncio.run_coroutine_threadsafe(_cancel_all(), _bg_loop)
        await asyncio.wait_for(asyncio.wrap_future(future), timeout=10)
        log.info("[Gateway] 后台常驻任务已优雅关闭")
    except asyncio.TimeoutError:
        log.warning("[Gateway] 后台常驻任务 10 秒内未完全收尾，放弃等待，进程继续正常退出")
    except Exception as e:
        log.error("[Gateway] 关闭后台任务时出错（不影响进程正常退出）: %s", e, exc_info=True)
 
 
def _set_telegram_menu_button():
    token        = os.environ.get("TG_BOT_TOKEN", "")
    gateway_host = os.environ.get("GATEWAY_HOST", "").rstrip("/")
    if not token or not gateway_host:
        return
    try:
        res = requests.post(
            f"https://api.telegram.org/bot{token}/setChatMenuButton",
            json={"menu_button": {
                "type": "web_app",
                "text": "config",
                "web_app": {"url": f"{gateway_host}/miniapp"},
            }},
            timeout=10,
        )
        data = res.json()
        if data.get("ok"):
            print(f"[OK] menu button set: {gateway_host}/miniapp")
    except Exception as e:
        log.error("[Gateway] 设置 menu button 异常: %s", e, exc_info=True)
 
 
def _set_telegram_webhook():
    import re as _re
    token        = os.environ.get("TG_BOT_TOKEN", "")
    gateway_host = os.environ.get("GATEWAY_HOST", "").rstrip("/")
    if not token or not gateway_host:
        log.warning("[Gateway] 未配置 TG_BOT_TOKEN 或 GATEWAY_HOST，跳过 webhook 注册")
        return
    try:
        payload = {
            "url": f"{gateway_host}/webhook",
            "allowed_updates": ["message"],
        }
        if API_SECRET:
            clean = _re.sub(r'[^A-Za-z0-9_\-]', '', API_SECRET)[:256]
            if clean:
                payload["secret_token"] = clean
        res = requests.post(
            f"https://api.telegram.org/bot{token}/setWebhook",
            json=payload,
            timeout=10,
        )
        data = res.json()
        if data.get("ok"):
            print(f"[OK] Webhook 已注册: {gateway_host}/webhook")
        else:
            log.warning("[Gateway] Webhook 注册失败: %s", data)
    except Exception as e:
        log.error("[Gateway] Webhook 注册异常: %s", e, exc_info=True)
 
 
if __name__ == "__main__":
    _set_telegram_menu_button()
    _set_telegram_webhook()
    start_background_tasks()
 
    from starlette.responses import JSONResponse
    from starlette.routing import Route
 
    def _admin_auth_ok(request) -> bool:
        """管理端点鉴权：复用 API_SECRET 作为 Bearer token
        （Authorization: Bearer $API_SECRET，与 /v1/chat/completions 相同）。
        未配置 API_SECRET 时跳过校验，行为与聊天端点保持一致。
        之前这三个端点完全无鉴权，URL 被扫到就能任意触发烧 LLM 额度的
        总结任务或重置 webhook。"""
        if not API_SECRET:
            return True
        auth = request.headers.get("authorization", "")
        token = auth.split(" ", 1)[-1].strip() if " " in auth else auth.strip()
        return token == API_SECRET
 
    async def trigger_summary(request):
        if not _admin_auth_ok(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        try:
            body = await request.body()
            params = {}
            if body:
                try:
                    params = json.loads(body)
                except Exception as e:
                    log.warning("[Gateway] /trigger-summary 请求体不是合法 JSON，按默认参数处理: %s", e)
 
            action = params.get("action", "nightly")
            _BEIJING = timezone(timedelta(hours=8))
 
            if action == "week":
                target_str = params.get("target_sunday", "")
                target = None
                if target_str:
                    from datetime import datetime as _dt
                    target = _dt.strptime(target_str, "%Y-%m-%d").replace(
                        hour=23, minute=59, second=59, tzinfo=_BEIJING)
                from scheduled import run_chat_week_summary
                await asyncio.to_thread(run_chat_week_summary, target)
                return JSONResponse({"ok": True, "message": f"周总结执行完成 target_sunday={target_str or '默认'}"})
 
            elif action == "month":
                target_str = params.get("target_month_end", "")
                target = None
                if target_str:
                    from datetime import datetime as _dt
                    target = _dt.strptime(target_str, "%Y-%m-%d").replace(
                        hour=23, minute=59, second=59, tzinfo=_BEIJING)
                from scheduled import run_chat_month_summary
                await asyncio.to_thread(run_chat_month_summary, target)
                return JSONResponse({"ok": True, "message": f"月总结执行完成 target_month_end={target_str or '默认'}"})
 
            else:
                from scheduled import run_nightly_summary
                await asyncio.to_thread(run_nightly_summary)
                return JSONResponse({"ok": True, "message": "凌晨总结已手动执行完成"})
        except Exception as e:
            log.error("[Gateway] /trigger-summary 处理异常: %s", e, exc_info=True)
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
 
    async def tg_status(request):
        if not _admin_auth_ok(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        token = os.environ.get("TG_BOT_TOKEN", "")
        if not token:
            return JSONResponse({"error": "TG_BOT_TOKEN not set"}, status_code=500)
        try:
            import httpx as _httpx
            async with _httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"https://api.telegram.org/bot{token}/getWebhookInfo")
                data = resp.json()
            return JSONResponse(data)
        except Exception as e:
            log.error("[Gateway] /tg-status 处理异常: %s", e, exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)
 
    async def tg_reset_webhook(request):
        if not _admin_auth_ok(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        import re as _re
        token        = os.environ.get("TG_BOT_TOKEN", "")
        gateway_host = os.environ.get("GATEWAY_HOST", "").rstrip("/")
        api_secret   = API_SECRET
        if not token or not gateway_host:
            return JSONResponse({"error": "TG_BOT_TOKEN 或 GATEWAY_HOST 未配置"}, status_code=500)
        try:
            import httpx as _httpx
            payload = {
                "url": f"{gateway_host}/webhook",
                "allowed_updates": ["message"],
            }
            if api_secret:
                clean = _re.sub(r'[^A-Za-z0-9_\-]', '', api_secret)[:256]
                if clean:
                    payload["secret_token"] = clean
            async with _httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    f"https://api.telegram.org/bot{token}/setWebhook",
                    json=payload,
                )
                data = resp.json()
            print(f"🔧 [tg-reset-webhook] 手动重新注册: {data}")
            return JSONResponse({"result": data, "webhook_url": f"{gateway_host}/webhook"})
        except Exception as e:
            log.error("[Gateway] /tg-reset-webhook 处理异常: %s", e, exc_info=True)
            return JSONResponse({"error": str(e)}, status_code=500)
 
    from contextlib import asynccontextmanager
 
    @asynccontextmanager
    async def _lifespan(app):
        # startup 阶段无需额外动作（start_background_tasks() 在更早之前已经
        # 同步调用过了），yield 之后到 shutdown 阶段再收尾后台任务。
        yield
        await shutdown_background_tasks()
 
    base_app = Starlette(routes=[
        Route("/trigger-summary", trigger_summary, methods=["POST"]),
        Route("/tg-status", tg_status, methods=["GET"]),
        Route("/tg-reset-webhook", tg_reset_webhook, methods=["GET"]),
    ], lifespan=_lifespan)
 
    app = RikkahubGatewayMiddleware(base_app)
    port = int(os.environ.get("PORT", 8000))
    print(f">> server started on port {port}")
 
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
 
    uvicorn.run(app, host="0.0.0.0", port=port, access_log=False)
