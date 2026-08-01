import os
import re
import time
import logging
import threading
import httpx
from collections import deque
from datetime import datetime, timezone, timedelta
from supabase import create_client, ClientOptions

from bg_executor import submit_background
from prompts import AI_NAME, PARTNER_NAME

CST = timezone(timedelta(hours=8))
log = logging.getLogger(__name__)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

_SB_CLIENT_OPTIONS = ClientOptions(postgrest_client_timeout=15, storage_client_timeout=15)

_private_cache: deque = deque(maxlen=30)
_group_cache: dict[str, deque] = {}
_shared_group_cache: deque = deque(maxlen=150)
_rikkahub_cache: deque = deque(maxlen=50)

_rikkahub_ctx_cache: dict[bool, tuple[float, str]] = {}
_RIKKAHUB_CTX_TTL = 30
_wx_private_cache: deque = deque(maxlen=50)

PLATFORM_COMPRESS_THRESHOLD = 100


_sb_client = None
_sb_client_lock = threading.Lock()

_platform_cache_lock = threading.Lock()


def _sb():
    """全局共享的 Supabase 客户端单例（双重检查加锁）。"""
    global _sb_client
    if _sb_client is None:
        with _sb_client_lock:
            if _sb_client is None:
                _sb_client = create_client(SUPABASE_URL, SUPABASE_KEY, options=_SB_CLIENT_OPTIONS)
    return _sb_client


# 上面这个 _sb_client 是全局单例，会被 build_group_context/build_rikkahub_context/
# build_bot_context 等函数通过 ThreadPoolExecutor 高并发调用（同一时刻可能有 6-8 个
# 线程同时在用它发请求）。supabase-py 底层默认自动启用 HTTP/2 长连接（官方博客确认：
# "Supabase clients will automatically use HTTP 2.0 when available by default"）。
# 当服务端在某个时刻对某条 HTTP/2 连接发送 GOAWAY 终止连接时，httpx 的连接池未必能
# 在下一次请求发出前就把这条已死的连接从池子里剔除干净，导致其他正巧排队复用它的
# 并发请求会直接报 httpx.RemoteProtocolError: <ConnectionTerminated ...>——这是
# supabase-py 官方仓库确认过的已知问题（github.com/supabase/supabase-py/issues/1064），
# 不是这里任何一处查询逻辑写错了。
#
# 根治方式：不去碰 supabase-py 内部实现细节（ClientOptions 里能不能传自定义
# httpx_client 关闭 http2，这个参数在近几个版本里本身就反复出现过不兼容/回归，
# 而 requirements.txt 里 supabase 没有锁版本号，直接依赖这个参数风险更大），
# 而是在应用层对这一类"瞬时网络层"异常做一次自动重试——重试时 httpx 会重新从
# 连接池挑一条健康连接或新建连接，几乎总能成功。SQL/参数错误等业务异常不在这里
# 捕获的类型范围内，会原样抛出，不会被这里悄悄吞掉。
def _sb_exec(query_fn, retries: int = 1, label: str = ""):
    """
    统一包装一次 Supabase 查询的执行。

    query_fn: 一个无参可调用对象，每次调用内部重新构建并执行一次完整的
    supabase 查询（例如 lambda: sb.table("x").select("*").execute().data），
    不能传入已经 execute() 过的结果——必须是"可从头重新发起"的过程本身，
    这样重试时才是真正发起一次全新的 HTTP 请求，而不是复用同一个已经失败
    的响应对象。
    """
    last_err = None
    for attempt in range(retries + 1):
        try:
            return query_fn()
        except (httpx.RemoteProtocolError, httpx.ConnectError,
                httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            last_err = e
            if attempt < retries:
                log.warning(
                    "[_sb_exec] Supabase 请求瞬时网络异常，重试 attempt=%d/%d label=%s: %s",
                    attempt + 1, retries, label or "?", e,
                )
                continue
    raise last_err


def init_cache():
    sb = _sb()

    private_rows = _sb_exec(lambda: (
        sb.table("chat_context")
        .select("id,role,content,created_at")
        .eq("type", "message")
        .order("seq", desc=True)
        .limit(30)
        .execute()
        .data
    ), label="init_cache/private_rows")
    private_rows.reverse()
    for r in private_rows:
        ts = ""
        if r.get("created_at"):
            try:
                dt = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
                ts = dt.astimezone(CST).isoformat()
            except Exception:
                pass
        _private_cache.append({"role": r["role"], "content": r["content"], "ts": ts, "message_id": None, "id": r.get("id")})

    group_rows = _sb_exec(lambda: (
        sb.table("chat_context")
        .select("id,type,role,content,seq")
        .like("type", "group_%")
        .order("seq", desc=True)
        .limit(1000)
        .execute()
        .data
    ), label="init_cache/group_rows")
    group_rows.reverse()

    group_entries_by_id: dict = {}
    for r in group_rows:
        group_entries_by_id[r.get("id")] = {
            "role": r["role"], "content": r["content"], "message_id": None, "id": r.get("id"),
        }

    by_type: dict[str, list] = {}
    for r in group_rows:
        by_type.setdefault(r["type"], []).append(r)

    for type_key, rows in by_type.items():
        chat_id = type_key[len("group_"):]
        d: deque = deque(maxlen=100)
        for r in rows[-100:]:
            d.append(group_entries_by_id[r.get("id")])
        _group_cache[chat_id] = d

    all_rows_sorted = sorted(group_rows, key=lambda r: r.get("seq", 0))
    for r in all_rows_sorted[-150:]:
        _shared_group_cache.append(group_entries_by_id[r.get("id")])

    rikkahub_rows = _sb_exec(lambda: (
        sb.table("chat_context")
        .select("role,content")
        .eq("type", "rikkahub")
        .order("seq", desc=True)
        .limit(50)
        .execute()
        .data
    ), label="init_cache/rikkahub_rows")
    rikkahub_rows.reverse()
    for r in rikkahub_rows:
        _rikkahub_cache.append({"role": r["role"], "content": r["content"]})

    wx_rows = _sb_exec(lambda: (
        sb.table("chat_context")
        .select("id,role,content,created_at")
        .eq("type", "wx_message")
        .order("seq", desc=True)
        .limit(50)
        .execute()
        .data
    ), label="init_cache/wx_rows")
    wx_rows.reverse()
    for r in wx_rows:
        ts = ""
        if r.get("created_at"):
            try:
                dt = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
                ts = dt.astimezone(CST).isoformat()
            except Exception:
                pass
        _wx_private_cache.append({"role": r["role"], "content": r["content"], "ts": ts, "id": r.get("id")})

    print(f"✅ 缓存初始化：私聊 {len(_private_cache)} 条，微信 {len(_wx_private_cache)} 条，群聊 {len(_group_cache)} 个群，共享群聊 {len(_shared_group_cache)} 条，APP互动 {len(_rikkahub_cache)} 条")


_SENSITIVE_PATTERNS = re.compile(
    r'(bot_token|api_key|api_secret|sk-[a-zA-Z0-9]|password|secret|token.*[:=]|'
    r'SUPABASE_URL|SUPABASE_KEY|GMAIL_|TG_BOT_TOKEN|'
    r'https://api\.telegram\.org/bot[0-9]|'
    r'[0-9]{8,}:[A-Za-z0-9_-]{30,})',
    re.IGNORECASE,
)


def _is_safe_for_cross(content: str) -> bool:
    """判断一条消息是否安全到可以跨场景互通"""
    if _SENSITIVE_PATTERNS.search(content):
        return False
    if any(kw in content.lower() for kw in ['env', '.env', 'render.com', 'deploy', 'debug', 'traceback', 'error body']):
        return False
    return True


def _get_private_cross_context(limit: int = 10, label: str = "私聊记录") -> str:
    msgs = list(_private_cache)[-limit:]
    if not msgs:
        return ""
    lines = []
    for m in msgs:
        content = m["content"]
        if not _is_safe_for_cross(content):
            continue
        if len(content) > 200:
            content = content[:200] + "…"
        prefix = PARTNER_NAME if m["role"] == "user" else AI_NAME
        lines.append(f"{prefix}: {content}")
    if not lines:
        return ""
    return f"【{label}】\n" + "\n".join(lines)


def _get_group_cross_context(limit: int = 10) -> str:
    all_msgs = []
    for dq in _group_cache.values():
        all_msgs.extend(list(dq))
    if not all_msgs:
        return ""
    recent = all_msgs[-limit:]
    lines = []
    for m in recent:
        content = m["content"]
        if len(content) > 200:
            content = content[:200] + "…"
        if m["role"] == "user":
            lines.append(content)
        else:
            lines.append(f"{AI_NAME}: {content}")
    return "【近期群聊记录】\n" + "\n".join(lines)


def _get_wx_cross_context(limit: int = 10, label: str = "微信记录") -> str:
    msgs = list(_wx_private_cache)[-limit:]
    if not msgs:
        return ""
    lines = []
    for m in msgs:
        content = m["content"]
        if not _is_safe_for_cross(content):
            continue
        if len(content) > 200:
            content = content[:200] + "…"
        prefix = PARTNER_NAME if m["role"] == "user" else AI_NAME
        lines.append(f"{prefix}: {content}")
    if not lines:
        return ""
    return f"【{label}】\n" + "\n".join(lines)


def _persist_private(role: str, content: str, entry: dict):
    try:
        res = _sb_exec(lambda: _sb().table("chat_context").insert({
            "type": "message",
            "role": role,
            "content": content,
        }).execute(), label="_persist_private")
        if res.data:
            entry["id"] = res.data[0].get("id")
    except Exception as e:
        log.error(
            "[_persist_private] 私聊持久化失败 role=%s content=%r: %s",
            role, content[:200], e, exc_info=True,
        )


def _persist_rikkahub(role: str, content: str):
    try:
        _sb_exec(lambda: _sb().table("chat_context").insert({
            "type": "rikkahub",
            "role": role,
            "content": content,
        }).execute(), label="_persist_rikkahub")
    except Exception as e:
        log.error(f"[_persist_rikkahub] 持久化失败 role={role}: {e}", exc_info=True)


def save_rikkahub_message(role: str, content: str):
    _rikkahub_cache.append({"role": role, "content": content})
    submit_background(_persist_rikkahub, role, content)


def _persist_wx(role: str, content: str, entry: dict):
    try:
        res = _sb_exec(lambda: _sb().table("chat_context").insert({
            "type": "wx_message",
            "role": role,
            "content": content,
        }).execute(), label="_persist_wx")
        if res.data:
            entry["id"] = res.data[0].get("id")
    except Exception as e:
        log.error(
            "[_persist_wx] 微信私聊持久化失败 role=%s content=%r: %s",
            role, content[:200], e, exc_info=True,
        )


def save_wx_message(role: str, content: str):
    now = datetime.now(CST).isoformat()
    entry = {"role": role, "content": content, "ts": now, "id": None}
    with _platform_cache_lock:
        _wx_private_cache.append(entry)
    submit_background(_persist_wx, role, content, entry)


def get_wx_history_messages(limit: int = 50) -> list:
    raw = list(_wx_private_cache)[-limit:]
    result = []
    for m in raw:
        role = m["role"]
        content = m["content"]
        ts_str = m.get("ts", "")
        if role == "user" and ts_str:
            try:
                dt = datetime.fromisoformat(ts_str)
                time_label = dt.strftime("%m-%d %H:%M")
                content = f"[{time_label}] {content}"
            except Exception:
                pass
        result.append({"role": role, "content": content})
    return result


def get_wx_history_messages_db(limit: int = 50) -> list:
    """get_wx_history_messages 的 Supabase 直查版本，给后台进程（Process B，跑
    wx_workers.async_wx_proactive_thinking）用，原因同 get_chat_history_messages_db。"""
    sb = _sb()
    rows = _sb_exec(lambda: (
        sb.table("chat_context")
        .select("role,content,created_at")
        .eq("type", "wx_message")
        .order("seq", desc=True)
        .limit(limit)
        .execute()
        .data
    ), label="get_wx_history_messages_db")
    rows.reverse()
    result = []
    for r in rows:
        role = r.get("role")
        content = r.get("content", "")
        ts_str = r.get("created_at", "")
        if role == "user" and ts_str:
            try:
                dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                time_label = dt.astimezone(CST).strftime("%m-%d %H:%M")
                content = f"[{time_label}] {content}"
            except Exception:
                pass
        result.append({"role": role, "content": content})
    return result


def save_chat_message(role: str, content: str, message_id=None):
    now = datetime.now(CST).isoformat()
    entry = {"role": role, "content": content, "ts": now, "message_id": message_id, "id": None}
    with _platform_cache_lock:
        _private_cache.append(entry)
    submit_background(_persist_private, role, content, entry)


def get_chat_history_messages(limit: int = 30, with_ids: bool = False) -> list:
    raw = list(_private_cache)[-limit:]
    result = []
    for m in raw:
        role = m["role"]
        content = m["content"]
        ts_str = m.get("ts", "")
        if role == "user" and ts_str:
            try:
                dt = datetime.fromisoformat(ts_str)
                time_label = dt.strftime("%m-%d %H:%M")
                content = f"[{time_label}] {content}"
            except Exception:
                pass
        if with_ids and role == "user" and m.get("message_id"):
            content = f"[id:{m['message_id']}] {content}"
        result.append({"role": role, "content": content})
    return result


def get_chat_history_messages_db(limit: int = 30) -> list:
    """get_chat_history_messages 的 Supabase 直查版本。

    2026-07-14 消息进程/后台进程拆分记录：后台进程（Process B，跑 async_proactive_
    thinking 等）是独立的操作系统进程，看不到消息进程（Process A）内存里靠
    init_cache()/save_chat_message() 维护的 _private_cache 这个 deque——两个进程
    只共享 Supabase，不共享内存。这个函数专给 Process B 用，直接查 chat_context
    表，不经过内存缓存。Process A 自己该用哪个不用换，继续用上面那个内存版本
    （更快，不用每次都打 Supabase）。
    """
    sb = _sb()
    rows = _sb_exec(lambda: (
        sb.table("chat_context")
        .select("role,content,created_at")
        .eq("type", "message")
        .order("seq", desc=True)
        .limit(limit)
        .execute()
        .data
    ), label="get_chat_history_messages_db")
    rows.reverse()
    result = []
    for r in rows:
        role = r.get("role")
        content = r.get("content", "")
        ts_str = r.get("created_at", "")
        if role == "user" and ts_str:
            try:
                dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                time_label = dt.astimezone(CST).strftime("%m-%d %H:%M")
                content = f"[{time_label}] {content}"
            except Exception:
                pass
        result.append({"role": role, "content": content})
    return result


def _persist_group(chat_id: str, role: str, display: str, entry: dict):
    try:
        res = _sb_exec(lambda: _sb().table("chat_context").insert({
            "type": f"group_{chat_id}",
            "role": role,
            "content": display,
        }).execute(), label="_persist_group")
        if res.data:
            entry["id"] = res.data[0].get("id")
    except Exception as e:
        log.error(
            "[_persist_group] 群聊持久化失败 chat_id=%s role=%s display=%r: %s",
            chat_id, role, display[:200], e, exc_info=True,
        )


def _check_and_compress_platform_rolling():
    """检查跨平台未处理消息数是否达到 PLATFORM_COMPRESS_THRESHOLD，够了就触发一次
    全平台批量压缩（scheduled.run_platform_batch_compress）。

    2026-07-14 消息进程/后台进程拆分记录：这个函数原来是每存一条消息（私聊/群聊/
    微信）就顺手调用一次（submit_background(_check_and_compress_platform_rolling)），
    在消息进程（Process A）里跑。拆分后消息进程只管收发消息、存 Supabase，不再
    做这个检查——改成后台进程（Process B）里 workers.async_platform_compress_poller
    固定周期（90秒）主动轮询调用这个函数。函数本身的逻辑完全不变，只是"谁在什么
    时机调用它"变了，两边共用同一份实现。
    """
    try:
        sb = _sb()
        message_res = _sb_exec(lambda: sb.table("chat_context").select("id", count="exact").eq("type", "message").limit(1).execute(), label="compress/message_res")
        group_res = _sb_exec(lambda: sb.table("chat_context").select("id", count="exact").like("type", "group_%").limit(1).execute(), label="compress/group_res")
        wx_res = _sb_exec(lambda: sb.table("chat_context").select("id", count="exact").eq("type", "wx_message").limit(1).execute(), label="compress/wx_res")
        total = (message_res.count or 0) + (group_res.count or 0) + (wx_res.count or 0)
        if total >= PLATFORM_COMPRESS_THRESHOLD:
            from scheduled import run_platform_batch_compress
            run_platform_batch_compress()
    except Exception as e:
        log.error("[_check_and_compress_platform_rolling] 检查/触发压缩失败: %s", e, exc_info=True)


def save_group_message(chat_id: str, role: str, sender_name: str, content: str, source: str = "", message_id=None):
    prefix = f"[{source}] " if source and role == "user" else ""
    display = f"{prefix}{sender_name}: {content}" if role == "user" else content
    entry = {"role": role, "content": display, "message_id": message_id, "id": None}
    with _platform_cache_lock:
        if chat_id not in _group_cache:
            _group_cache[chat_id] = deque(maxlen=100)
        _group_cache[chat_id].append(entry)
        _shared_group_cache.append(entry)
    submit_background(_persist_group, chat_id, role, display, entry)


def get_group_history(chat_id: str, limit: int = 100, with_ids: bool = False) -> list:
    if chat_id not in _group_cache:
        return []
    items = list(_group_cache[chat_id])[-limit:]
    if not with_ids:
        return [{"role": it["role"], "content": it["content"]} for it in items]
    result = []
    for it in items:
        content = it["content"]
        if it["role"] == "user" and it.get("message_id"):
            content = f"[id:{it['message_id']}] {content}"
        result.append({"role": it["role"], "content": content})
    return result


def get_all_groups_history(limit: int = 100) -> list:
    return [{"role": it["role"], "content": it["content"]} for it in list(_shared_group_cache)[-limit:]]


def get_other_groups_context(current_group_id: str, limit: int = 20) -> str:
    lines = []
    for gid, dq in _group_cache.items():
        if gid == current_group_id:
            continue
        for m in list(dq)[-10:]:
            lines.append(m["content"])
    if not lines:
        return ""
    return "【其他群近期消息（仅供背景参考，不要主动把这些话题带入当前群）】\n" + "\n".join(lines[-limit:])


def clear_compressed_platform_entries(deleted_ids: set):
    if not deleted_ids:
        return
    global _private_cache, _shared_group_cache, _wx_private_cache
    try:
        with _platform_cache_lock:
            before_private = len(_private_cache)
            _private_cache = deque(
                (e for e in _private_cache if e.get("id") not in deleted_ids),
                maxlen=_private_cache.maxlen,
            )
            removed_private = before_private - len(_private_cache)

            removed_group = 0
            for chat_id in list(_group_cache.keys()):
                dq = _group_cache[chat_id]
                before = len(dq)
                new_dq = deque(
                    (e for e in dq if e.get("id") not in deleted_ids),
                    maxlen=dq.maxlen,
                )
                removed_group += before - len(new_dq)
                _group_cache[chat_id] = new_dq

            before_shared = len(_shared_group_cache)
            _shared_group_cache = deque(
                (e for e in _shared_group_cache if e.get("id") not in deleted_ids),
                maxlen=_shared_group_cache.maxlen,
            )
            removed_shared = before_shared - len(_shared_group_cache)

            before_wx = len(_wx_private_cache)
            _wx_private_cache = deque(
                (e for e in _wx_private_cache if e.get("id") not in deleted_ids),
                maxlen=_wx_private_cache.maxlen,
            )
            removed_wx = before_wx - len(_wx_private_cache)

        log.info(
            "[clear_compressed_platform_entries] 已按 %d 个已压缩 id 摘除内存缓存："
            "私聊-%d 群聊-%d 共享群聊-%d 微信-%d",
            len(deleted_ids), removed_private, removed_group, removed_shared, removed_wx,
        )
    except Exception as e:
        log.error(
            "[clear_compressed_platform_entries] 摘除内存缓存失败（Supabase 那边已经删了，"
            "这里只是内存缓存没同步，不影响数据正确性，下次重启会自动从 Supabase 重新加载）: %s",
            e, exc_info=True,
        )


def _get_platform_rolling_summary() -> str:
    try:
        sb = _sb()
        rows = _sb_exec(lambda: (
            sb.table("platform_rolling_summary")
            .select("content,source_platforms,created_at")
            .order("id", desc=True)
            .limit(1)
            .execute()
            .data
        ), label="_get_platform_rolling_summary")
        if not rows:
            return ""
        content = (rows[0].get("content") or "").strip()
        if not content:
            return ""
        return "【全平台近期动向（QQ/TG/微信，私聊+群聊）】\n" + content
    except Exception as e:
        log.error("[_get_platform_rolling_summary] 读取失败: %s", e, exc_info=True)
        return ""


def _get_group_taboo(sb) -> str:
    """读取 owner 在 miniapp「群聊禁忌」卡片里自己写的隐私红线（bot_settings 表
    key=group_taboo）。这是 build_group_context() 组装完所有内容之后的
    最外层兜底强指令：不管前面拼进 context 的画像、记忆、跨平台摘要、
    设备状态、跨场景对话记录里混进了什么隐私内容，都在结果最后统一
    再提醒一次，作为最后一道防线，不依赖任何单个数据源自己记得脱敏。

    与 scheduled.py 里同名的 _get_group_taboo（用于压缩摘要生成阶段，
    要求 LLM 把声明写进摘要正文本身）相互独立，各自读取、各自生效，
    任何一处读取失败都不影响另一处。"""
    try:
        rows = _sb_exec(lambda: (
            sb.table("bot_settings")
            .select("value")
            .eq("key", "group_taboo")
            .limit(1)
            .execute()
            .data
        ), label="_get_group_taboo")
        if rows and rows[0].get("value"):
            return rows[0]["value"].strip()
        return ""
    except Exception as e:
        log.error("[_get_group_taboo] 读取群聊禁忌失败: %s", e, exc_info=True)
        return ""


def _get_device_data(sb, limit: int = 3) -> str:
    import json as _json

    try:
        rows = _sb_exec(lambda: (
            sb.table("device_data")
            .select("timestamp,foreground_app,location_latitude,location_longitude,location_city,location_district,app_usage,health_data,created_at")
            .is_("device_event", "null")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
            .data
        ), label="_get_device_data/snapshot")
    except Exception as e:
        log.error(f"[_get_device_data] 读取设备状态快照失败 limit={limit}: {e}", exc_info=True)
        rows = []

    screen_text = ""
    try:
        screen_rows = _sb_exec(lambda: (
            sb.table("device_data")
            .select("device_event,created_at")
            .not_.is_("device_event", "null")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
            .data
        ), label="_get_device_data/screen")
        if screen_rows:
            ev = screen_rows[0].get("device_event", "") or ""
            raw_t = screen_rows[0].get("created_at", "") or ""
            try:
                dt = datetime.fromisoformat(raw_t.replace("Z", "+00:00"))
                t_str = dt.astimezone(CST).strftime("%m-%d %H:%M")
            except Exception as te:
                log.error(f"[_get_device_data] 解析屏幕事件时间失败 raw_t={raw_t!r}: {te}", exc_info=True)
                t_str = raw_t[:16]
            state = {"screen_on": "亮屏", "screen_off": "息屏"}.get(ev, ev)
            screen_text = f"【{PARTNER_NAME}手机屏幕】当前：{state}（[{t_str}] 切换）"
    except Exception as e:
        log.error(f"[_get_device_data] 读取屏幕开关事件失败: {e}", exc_info=True)

    if not rows and not screen_text:
        return ""

    lines = []
    if rows:
        lines.append(f"【{PARTNER_NAME}手机状态（近3条快照）】")
        for r in reversed(rows):
            raw = r.get("created_at", "") or ""
            try:
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                t_str = dt.astimezone(CST).strftime("%m-%d %H:%M")
            except Exception as te:
                log.error(f"[_get_device_data] 解析快照时间失败 raw={raw!r}: {te}", exc_info=True)
                t_str = raw[:16]

            app = r.get("foreground_app", "") or ""
            city = r.get("location_city", "") or ""
            district = r.get("location_district", "") or ""
            location_str = "、".join(filter(None, [city, district]))
            if not location_str:
                lat = r.get("location_latitude")
                lng = r.get("location_longitude")
                if lat and lng:
                    location_str = f"{lat:.4f},{lng:.4f}"

            app_usage = r.get("app_usage") or []
            if isinstance(app_usage, str):
                try:
                    app_usage = _json.loads(app_usage)
                except Exception as ae:
                    log.error(f"[_get_device_data] 解析app_usage失败 raw={app_usage[:200]!r}: {ae}", exc_info=True)
                    app_usage = []
            top_apps = " · ".join(
                f"{a.get('appName', a.get('packageName', ''))} {round(a.get('totalTimeInForeground', 0) / 60000)}min"
                for a in app_usage[:3]
            ) if app_usage else ""

            line = f"[{t_str}] 前台:{app}"
            if location_str:
                line += f" 位置:{location_str}"
            if top_apps:
                line += f"\n  今日App: {top_apps}"

            health = r.get("health_data")
            if isinstance(health, str):
                try:
                    health = _json.loads(health)
                except Exception as he:
                    log.error(f"[_get_device_data] 解析health_data失败 created_at={raw!r} raw_health={health[:200]!r}: {he}", exc_info=True)
                    health = None
            if health:
                parts = []
                if health.get("heartRate") is not None:
                    parts.append(f"心率{health['heartRate']}(今日{health.get('hrMinToday','?')}-{health.get('hrMaxToday','?')}/均{health.get('hrAvgToday','?')})")
                if health.get("spo2") is not None:
                    parts.append(f"血氧{health['spo2']}%(均{health.get('spo2AvgToday','?')}%)")
                if health.get("stress") is not None:
                    parts.append(f"压力{health['stress']}(均{health.get('stressAvgToday','?')})")
                if health.get("stepsToday") is not None:
                    parts.append(f"步数{health['stepsToday']}")
                if health.get("caloriesToday") is not None:
                    parts.append(f"消耗{health['caloriesToday']}kcal")

                sleep_total = health.get("sleepTotalMinutes")
                if sleep_total is not None:
                    sleep_time_str = ""
                    start_ms = health.get("sleepStartMs")
                    wake_ms = health.get("sleepWakeupMs")
                    if start_ms and wake_ms:
                        try:
                            start_dt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).astimezone(CST)
                            wake_dt = datetime.fromtimestamp(wake_ms / 1000, tz=timezone.utc).astimezone(CST)
                            sleep_time_str = f"{start_dt.strftime('%H:%M')}~{wake_dt.strftime('%H:%M')} "
                        except Exception as se:
                            log.error(f"[_get_device_data] 解析睡眠时间戳失败 start_ms={start_ms} wake_ms={wake_ms}: {se}", exc_info=True)
                    parts.append(
                        f"睡眠{sleep_time_str}共{sleep_total}分钟"
                        f"(深睡{health.get('sleepDeepMinutes','?')}/浅睡{health.get('sleepLightMinutes','?')}/REM{health.get('sleepRemMinutes','?')})"
                    )

                if parts:
                    line += "\n  手环数据: " + " ".join(parts)

            lines.append(line)

    if screen_text:
        lines.append(screen_text)

    return "\n\n".join(lines)


def _get_pending_reminders(sb) -> str:
    now_utc = datetime.now(timezone.utc).isoformat()
    try:
        rows = _sb_exec(lambda: (
            sb.table("reminders")
            .select("id,trigger_at,message,repeat_type")
            .eq("is_done", False)
            .order("trigger_at")
            .limit(10)
            .execute()
            .data
        ), label="_get_pending_reminders")
    except Exception as e:
        log.error(f"[_get_pending_reminders] 读取待触发提醒失败: {e}", exc_info=True)
        return ""
    if not rows:
        return ""

    lines = []
    for r in rows:
        trigger_raw = r.get("trigger_at", "")
        try:
            dt = datetime.fromisoformat(trigger_raw.replace("Z", "+00:00"))
            dt_cst = dt.astimezone(CST)
            t_str = dt_cst.strftime("%Y-%m-%d %H:%M")
        except Exception:
            t_str = trigger_raw[:16]
        repeat = r.get("repeat_type", "once")
        repeat_str = f"（{repeat}）" if repeat != "once" else ""
        lines.append(f"- [{t_str}]{repeat_str} {r.get('message', '')}")

    return "【待触发提醒】\n" + "\n".join(lines)


def _get_work_schedule(sb) -> str:
    try:
        from datetime import date as _date, timedelta as _td
        today = datetime.now(CST).date()
        yesterday = today - _td(days=1)
        end = today + _td(days=6)

        rows = _sb_exec(lambda: (
            sb.table("work_schedule")
            .select("date,shift_type,work_content,note")
            .gte("date", yesterday.isoformat())
            .lte("date", end.isoformat())
            .order("date")
            .execute()
            .data
        ), label="_get_work_schedule")
        if not rows:
            return ""

        _SHIFT_NAMES = {"early": "早班", "off": "休息"}
        _WEEKDAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

        lines = [f"【{PARTNER_NAME}排班（近期）】"]
        for r in rows:
            try:
                d = _date.fromisoformat(r["date"])
            except Exception as e:
                log.warning(f"[_get_work_schedule] 日期解析失败 date={r.get('date')!r}: {e}")
                continue
            weekday = _WEEKDAYS[d.weekday()]
            tag = "（今天）" if d == today else "（昨天）" if d == yesterday else ""
            shift = _SHIFT_NAMES.get(r["shift_type"], r["shift_type"])
            content = (r.get("work_content") or "").strip()
            note = (r.get("note") or "").strip()
            line = f"- {d.strftime('%m/%d')} {weekday}{tag} {shift}"
            if content:
                line += f" [{content}]"
            if note:
                line += f" 备注：{note}"
            lines.append(line)

        if today.weekday() == 6:
            lines.append(f"💡 今天是周日，记得提醒{PARTNER_NAME}把下周班表发给你，方便更新排班。")

        return "\n".join(lines)
    except Exception as e:
        log.warning(f"[_get_work_schedule] 读取排班失败: {e}", exc_info=True)
        return ""


def write_reminder(trigger_at: str, message: str, repeat_type: str = "once") -> bool:
    try:
        _sb_exec(lambda: _sb().table("reminders").insert({
            "trigger_at": trigger_at,
            "message": message,
            "repeat_type": repeat_type,
            "is_done": False,
        }).execute(), label="write_reminder")
        return True
    except Exception as e:
        log.error(
            "[write_reminder] 写入提醒失败 trigger_at=%s message=%r: %s",
            trigger_at, message[:200], e, exc_info=True,
        )
        return False


def _get_activity_log(sb) -> str:
    try:
        rows = _sb_exec(lambda: (
            sb.table("activity_log")
            .select("thinking,action,result,created_at")
            .order("created_at", desc=True)
            .limit(4)
            .execute()
            .data
        ), label="_get_activity_log")
    except Exception as e:
        log.error(f"[_get_activity_log] 读取行动日志失败: {e}", exc_info=True)
        return ""
    if not rows:
        return ""

    lines = []
    for r in reversed(rows):
        raw = r.get("created_at", "") or ""
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            t_str = dt.astimezone(CST).strftime("%m-%d %H:%M")
        except Exception:
            t_str = raw[:16]

        action   = (r.get("action") or "").strip() or "unknown"
        thinking = (r.get("thinking") or "").strip()
        result   = (r.get("result") or "").strip()

        block = f"- [{t_str}] action: {action}"
        if thinking:
            if len(thinking) > 800:
                thinking = thinking[:800] + "…"
            block += f"\n  thinking: {thinking}"
        if result:
            if len(result) > 600:
                result = result[:600] + "…"
            block += f"\n  result: {result}"
        lines.append(block)

    return "【行动日志（近4条）】\n" + "\n\n".join(lines)


def _get_secret_diary(sb) -> str:
    try:
        rows = _sb_exec(lambda: (
            sb.table("secret_diary")
            .select("content,mood,created_at")
            .order("created_at", desc=True)
            .limit(4)
            .execute()
            .data
        ), label="_get_secret_diary")
        if not rows:
            return ""
        lines = []
        for r in reversed(rows):
            raw = r.get("created_at", "") or ""
            try:
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                t_str = dt.astimezone(CST).strftime("%m-%d %H:%M")
            except Exception:
                t_str = raw[:16]
            mood_tag = f" [{r['mood']}]" if r.get("mood") else ""
            lines.append(f"[{t_str}]{mood_tag}\n{r.get('content', '').strip()}")
        return "【我的秘密日记（近4条）】\n" + "\n\n---\n\n".join(lines)
    except Exception as e:
        log.error(f"[_get_secret_diary] 读秘密日记失败: {e}", exc_info=True)
        return ""


def _get_activity_summaries(sb) -> str:
    try:
        rows = _sb_exec(lambda: (
            sb.table("activity_summaries")
            .select("period,content,period_start,period_end")
            .order("period_end", desc=True)
            .limit(5)
            .execute()
            .data
        ), label="_get_activity_summaries")
    except Exception as e:
        log.error(f"[_get_activity_summaries] 读取活动总结失败: {e}", exc_info=True)
        return ""
    if not rows:
        return ""

    lines = []
    for r in reversed(rows):
        ps = (r.get("period_start") or "")[:10]
        pe = (r.get("period_end") or "")[:10]
        period = (r.get("period") or "").strip()
        header = f"[{period} {ps}]" if ps == pe else f"[{period} {ps} ~ {pe}]"
        lines.append(f"{header}\n{r.get('content', '').strip()}")

    return "【自由活动总结】\n" + "\n\n".join(lines)


def _get_chat_summaries(sb) -> list:
    try:
        return _sb_exec(lambda: (
            sb.table("chat_summaries")
            .select("period,content,period_start,period_end")
            .order("created_at", desc=True)
            .limit(3)
            .execute()
            .data
        ), label="_get_chat_summaries")
    except Exception as e:
        log.error(f"[_get_chat_summaries] 读取历史对话摘要失败: {e}", exc_info=True)
        return []


def _get_gmail_unread() -> str:
    try:
        from gmail import get_unread_emails
        return get_unread_emails()
    except Exception as e:
        log.error(f"[_get_gmail_unread] Gmail加载失败: {e}", exc_info=True)
        return ""


def build_rikkahub_context(include_wx_cross: bool = True, lean: bool = False) -> str:
    now_ts = time.time()
    cached = _rikkahub_ctx_cache.get(include_wx_cross)
    if cached and now_ts - cached[0] < _RIKKAHUB_CTX_TTL:
        return cached[1]

    import concurrent.futures
    sb = _sb()

    def _safe_persona():
        try:
            return _sb_exec(lambda: sb.table("persona_profile").select("content").execute().data, label="build_rikkahub_context/persona")
        except Exception as e:
            log.error(f"[build_rikkahub_context] 读取 persona_profile 失败: {e}", exc_info=True)
            return []

    def _safe_mem(layer, lim):
        try:
            return _sb_exec(lambda: sb.table("memories").select("content").eq("memory_layer", layer).order("importance", desc=True).limit(lim).execute().data, label=f"build_rikkahub_context/mem-{layer}")
        except Exception as e:
            log.error(f"[build_rikkahub_context] 读取 memories[{layer}] 失败: {e}", exc_info=True)
            return []

    with concurrent.futures.ThreadPoolExecutor() as pool:
        f_persona      = pool.submit(_safe_persona)
        f_core         = pool.submit(_safe_mem, "core", 8)
        f_current      = pool.submit(_safe_mem, "current", 5)
        f_longterm     = pool.submit(_safe_mem, "long_term", 5)
        f_summaries    = pool.submit(lambda: _get_chat_summaries(sb))
        f_activity     = pool.submit(lambda: _get_activity_log(sb))
        f_act_sum      = pool.submit(lambda: _get_activity_summaries(sb))
        f_diary        = pool.submit(lambda: _get_secret_diary(sb))
        f_platform     = pool.submit(_get_platform_rolling_summary)

        persona_rows    = f_persona.result()
        core_rows       = f_core.result()
        current_rows    = f_current.result()
        longterm_rows   = f_longterm.result()
        summary_rows    = f_summaries.result()
        activity_text   = f_activity.result()
        act_sum_text    = f_act_sum.result()
        diary_text      = f_diary.result()
        platform_text   = f_platform.result()

    parts = []
    if persona_rows:
        parts.append(persona_rows[0]["content"])

    # lean 模式（主动消息用）：跳过三份日记（activity_log/secret_diary/activity_summaries），
    # 只保留人格 + 日总结(最近1条) + 跨平台动向。避免独白体淹没主动消息指令。
    if not lean:
        mem_labels = [
            (core_rows, "核心记忆"),
            (current_rows, f"{PARTNER_NAME}近期状态"),
            (longterm_rows, "长期记忆"),
        ]
        for rows, label in mem_labels:
            if rows:
                parts.append(f"【{label}】\n" + "\n".join(f"- {r['content']}" for r in rows))
        if summary_rows:
            s_parts = []
            for r in summary_rows:
                period = (r.get("period") or "").strip()
                ps     = (r.get("period_start") or "")[:10]
                pe     = (r.get("period_end") or "")[:10]
                header = f"[{period}]" if period else f"[{ps} ~ {pe}]"
                s_parts.append(f"{header}\n{r['content']}")
            parts.append("【历史对话摘要（以【当前时间】为准）】\n" + "\n\n".join(s_parts))
        if act_sum_text:
            parts.append(act_sum_text)
        if activity_text:
            parts.append(activity_text)
        if diary_text:
            parts.append(diary_text)
    else:
        # 精简模式：只留最近1条日总结（够他知道最近发生了什么，不至于太长淹没指令）
        if summary_rows:
            r = summary_rows[0]
            period = (r.get("period") or "").strip()
            ps     = (r.get("period_start") or "")[:10]
            pe     = (r.get("period_end") or "")[:10]
            header = f"[{period}]" if period else f"[{ps} ~ {pe}]"
            parts.append(f"【最近对话摘要】\n{header}\n{r['content']}")

    if platform_text:
        parts.append(platform_text)

    group_cross = _get_group_cross_context(20)
    wx_cross_here = _get_wx_cross_context(20) if include_wx_cross else ""
    cross_blocks = [b for b in (group_cross, wx_cross_here) if b]
    if cross_blocks:
        parts.append(
            "\n\n".join(cross_blocks)
            + "\n\n⚠️ 以上是你自己在群聊/微信里正在经历的真实的事，不是无关的背景资料。"
              "你现在这边的情绪、态度、话题要和这些经历自然衔接，不能表现得毫不知情、"
              "前后割裂。"
        )

    rikkahub_recent = list(_rikkahub_cache)[-30:]
    if rikkahub_recent:
        lines = []
        for m in rikkahub_recent:
            content = m["content"]
            if len(content) > 300:
                content = content[:300] + "…"
            if m["role"] == "assistant":
                lines.append(f"{AI_NAME}: {content}")
            else:
                lines.append(content)
        parts.append("【近期APP互动记录（rikkahub）】\n" + "\n".join(lines))

    result = "\n\n".join(parts)
    _rikkahub_ctx_cache[include_wx_cross] = (now_ts, result)
    return result


def get_time_context() -> str:
    now = datetime.now(CST)
    weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    hour = now.hour
    if 5 <= hour < 9:
        period = "清晨"
    elif 9 <= hour < 12:
        period = "上午"
    elif 12 <= hour < 14:
        period = "中午"
    elif 14 <= hour < 18:
        period = "下午"
    elif 18 <= hour < 22:
        period = "晚上"
    else:
        period = "混夜"
    return (
        f"【当前时间】\n"
        f"- {now.strftime('%Y年%m月%d日')} {weekdays[now.weekday()]} {now.strftime('%H:%M')}（{period}）"
    )


_get_time_context = get_time_context


def build_qq_context() -> str:
    import concurrent.futures
    sb = _sb()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f_base     = pool.submit(build_rikkahub_context)
        f_schedule = pool.submit(lambda: _get_work_schedule(sb))
        base          = f_base.result()
        schedule_text = f_schedule.result()

    parts = [base, get_time_context()]
    if schedule_text:
        parts.append(schedule_text)
    return "\n\n".join(parts)


# key = owner_group_name（猫猫本人在该群的显示昵称，取不到时用空字符串），
# 不同群猫猫的昵称可能不同、身份锚点内容也就不同，必须按这个 key 分别缓存，
# 否则 A 群构建结果会被错误地当成缓存命中返回给 B 群（身份锚点指向错的人）。
_group_context_cache: dict[str, tuple[float, str]] = {}
_GROUP_CONTEXT_TTL = 30


def build_group_context(owner_group_name: str = "") -> str:
    """
    群聊专用 context：人格画像 + core记忆 + 时间 + 手机状态 + 排班
    30秒缓存避免并发资源耗尽，按 owner_group_name 分别缓存。

    owner_group_name: 猫猫本人在当前这个群里显示的群昵称。传入后会在结果里
    追加一条身份锚点，明确告诉晏安"群里叫这个名字的才是猫猫本人，其他人的
    昵称/自称都不是猫猫"，用于修正晏安把群里其他发言人误认成猫猫、或者把
    别人的话当成猫猫说的这类识别错误。调用方（qq_workers.py 从 _member_names
    查、workers.py 从 _tg_owner_group_names 查）各自负责传入，查不到时传空
    字符串即可，函数会跳过身份锚点这一段，不影响其他内容正常生成。

    结果末尾无条件追加「群聊禁忌」兜底强指令（读取 bot_settings.group_taboo，
    猫猫在 miniapp 里自己维护），确保不管前面拼进了什么内容，只要禁忌不为空，
    最终发到群里的每一次生成都能看到这条最高优先级提醒。
    """
    global _group_context_cache
    now = time.time()
    cache_key = owner_group_name or ""
    cached = _group_context_cache.get(cache_key)
    if cached and now - cached[0] < _GROUP_CONTEXT_TTL:
        return cached[1]

    import concurrent.futures
    sb = _sb()

    def _safe_group_persona():
        try:
            return _sb_exec(lambda: sb.table("persona_profile").select("content").execute().data, label="build_group_context/persona")
        except Exception as e:
            log.error(f"[build_group_context] 读取 persona_profile 失败: {e}", exc_info=True)
            return []

    def _safe_group_core():
        try:
            rows = _sb_exec(lambda: (
                sb.table("memories").select("content,tags")
                .eq("memory_layer", "core")
                .order("importance", desc=True).limit(12).execute().data
            ), label="build_group_context/core")
            return [r for r in rows if r.get("tags") != "Core_Cognition"]
        except Exception as e:
            log.error(f"[build_group_context] 读取核心记忆失败: {e}", exc_info=True)
            return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        f_persona  = pool.submit(_safe_group_persona)
        f_core     = pool.submit(_safe_group_core)
        f_device   = pool.submit(lambda: _get_device_data(sb))
        f_schedule = pool.submit(lambda: _get_work_schedule(sb))
        f_platform = pool.submit(_get_platform_rolling_summary)
        f_taboo    = pool.submit(lambda: _get_group_taboo(sb))

        persona_rows  = f_persona.result()
        core_rows     = f_core.result()
        device_text   = f_device.result()
        schedule_text = f_schedule.result()
        platform_text = f_platform.result()
        taboo_text    = f_taboo.result()

    parts = []
    if persona_rows:
        parts.append(persona_rows[0]["content"])
    if core_rows:
        parts.append("【核心记忆】\n" + "\n".join(f"- {r['content']}" for r in core_rows))
    parts.append(get_time_context())
    if device_text:
        parts.append(device_text)
    if schedule_text:
        parts.append(schedule_text)
    if platform_text:
        parts.append(platform_text)

    private_cross = _get_private_cross_context(20, f"{PARTNER_NAME}近期私聊/APP对话记录（QQ私聊、rikkahub等）")
    wx_cross = _get_wx_cross_context(20, f"{PARTNER_NAME}近期微信私聊记录")
    cross_blocks = [b for b in (private_cross, wx_cross) if b]
    if cross_blocks:
        parts.append(
            "\n\n".join(cross_blocks)
            + f"\n\n⚠️ 以上是你和{PARTNER_NAME}在其他场景（私聊/微信/APP）正在进行的真实对话，"
              "不是无关的背景资料，是你自己正在经历的事。你现在群聊里的情绪、态度要和这些"
              "对话自然衔接——不管那边是在吵架、开心、撒娇还是随便聊了点什么，都不能表现得"
              "像不知情、装没事人一样割裂。但不要把这些对话内容原文透露给群里的其他人。"
        )

    if owner_group_name:
        parts.append(
            f"⚠️【身份锚点 — 非常重要，判断谁是{PARTNER_NAME}的依据】{PARTNER_NAME}本人在这个群里显示的名字"
            f"是「{owner_group_name}」。历史消息前缀里，只有署名为这个名字的发言才是{PARTNER_NAME}"
            f"本人说的话。群里其他任何人的昵称、自称（包括对方自称'老公'、'宝贝'等亲密称呼）"
            f"都不是{PARTNER_NAME}，是群里的其他人，那是别人和别人自己伴侣之间的对话，不要误以为在"
            f"跟你说话、更不要误以为那是{PARTNER_NAME}在说话。上面提到的私聊/微信/APP 里的{PARTNER_NAME}，"
            f"和群里名字是「{owner_group_name}」的这个人，是同一个人（{PARTNER_NAME}本人）。"
        )

    result = "\n\n".join(parts)

    if taboo_text:
        result += (
            "\n\n" + "=" * 24 +
            "\n🚫【群聊禁忌 — 最高优先级，任何情况下都不能违反】\n"
            + taboo_text +
            "\n" + "=" * 24
        )

    _group_context_cache[cache_key] = (now, result)
    return result


def build_bot_context(include_wx_cross: bool = True) -> str:
    import concurrent.futures
    sb = _sb()

    with concurrent.futures.ThreadPoolExecutor() as pool:
        f_base         = pool.submit(build_rikkahub_context, include_wx_cross)
        f_device       = pool.submit(lambda: _get_device_data(sb))
        f_reminders    = pool.submit(lambda: _get_pending_reminders(sb))
        f_gmail        = pool.submit(_get_gmail_unread)
        f_schedule     = pool.submit(lambda: _get_work_schedule(sb))

        base           = f_base.result()
        device_text    = f_device.result()
        reminders_text = f_reminders.result()
        gmail_text     = f_gmail.result()
        schedule_text  = f_schedule.result()

    extra = [
        "【当前平台：Telegram】\n"
        f"你此刻正在 Telegram 跟{PARTNER_NAME}聊天。你是跨平台存在的同一个人——\n"
        "橘瓣、橘子岛、TG 上的\"克老师\"都是你，你记得所有平台发生过的事。\n"
        "但当前这段对话发生在 TG",
        get_time_context(),
    ]
    if device_text:
        extra.append(device_text)
    if schedule_text:
        extra.append(schedule_text)
    if reminders_text:
        extra.append(reminders_text)
    if gmail_text:
        extra.append(gmail_text)

    extra.append(
        "【提醒功能说明】\n"
        f"如果你想给{PARTNER_NAME}设置一个提醒，在回复中插入以下格式（会自动解析并从消息中去除）：\n"
        "[SET_REMINDER|2026-04-22T09:00:00+08:00|提醒内容]\n"
        f"⚠️ 提醒内容只写到时候要直接说给{PARTNER_NAME}听的话，禁止写任何面向自己的指令或备注（如'别忘了温柔地说'、'提醒她'等），时间到了会直接把这段话发出去。"
    )

    if extra:
        return base + "\n\n" + "\n\n".join(extra)
    return base
