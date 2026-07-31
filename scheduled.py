import os
import json
import logging
import threading
import requests
from datetime import datetime, timedelta, timezone
 
import prompts
from prompts import AI_NAME, PARTNER_NAME
 
log = logging.getLogger(__name__)
 
BEIJING = timezone(timedelta(hours=8))
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
 
_SB_HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation",
}
 
_platform_compress_lock = threading.Lock()
 
 
def _clean_surrogates(text: str) -> str:
    return text.encode("utf-8", errors="ignore").decode("utf-8")
 
 
def _get_llm_config() -> dict:
    """全平台压缩、日/周/月/年总结、人格反思等所有后台任务专用的 LLM 配置。

    2026-07-14 排查记录：这里之前读的是 llm_config 表 active=eq.true 那一行，
    跟 utils.get_active_llm_config()（主对话用）读的是同一行——也就是后台任务
    和实时聊天共用同一个供应商账号。全平台批量压缩（run_platform_batch_compress）
    在群聊活跃时平均每5~18分钟就触发一次，每次都是一次完整的LLM请求，
    长期占用主对话那个账号的并发/速率配额，是"群聊回复持续变慢"的实测根因
    （用数据库时间戳核实过触发频率，不是猜测）。

    现在改成读 bg_active=eq.true 这一行，物理隔离成独立的供应商/账号，
    跟主对话彻底不共享配额。fallback 环境变量也换成独立的 BG_* 前缀，
    避免 Supabase 查询失败这种极端情况下又退回去和主对话共用同一套
    CHAT_*/BOT_MODEL 环境变量、白隔离一场。
    """
    try:
        res = requests.get(
            f"{SUPABASE_URL}/rest/v1/llm_config?bg_active=eq.true&limit=1",
            headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"},
            timeout=5,
        )
        data = res.json()
        if data:
            cfg = {
                "base_url": data[0]["base_url"].rstrip("/"),
                "api_key": data[0]["api_key"],
                "model": data[0]["model"],
                "extra_headers": data[0].get("extra_headers") or {},
            }
            log.info(
                "[_get_llm_config] 后台任务使用独立压缩模型 name=%s model=%s",
                data[0].get("name", "?"), cfg["model"],
            )
            return cfg
        log.warning("[_get_llm_config] llm_config 表里没有 bg_active=true 的行，回退到 BG_* 环境变量")
    except Exception as e:
        log.warning(f"读取 llm_config(bg_active) 失败，回退到 BG_* 环境变量: {e}")
    return {
        "base_url": os.environ.get("BG_CHAT_BASE_URL", "https://relay-cache.sharkielab.com/v1").rstrip("/"),
        "api_key": os.environ.get("BG_CHAT_API_KEY", os.environ.get("CHAT_API_KEY", "")),
        "model": os.environ.get("BG_BOT_MODEL", "[特特价次kiro]claude-opus-4-6-thinking"),
        "extra_headers": {},
    }
 
 
def _llm_chat(system: str, user: str, max_tokens: int = 1000, retries: int = 3) -> str:
    """非流式 LLM 调用，失败自动重试。
 
    失败时会把 HTTP 状态码、接口返回原文（或网络异常堆栈）完整记录到日志，
    而不只是 Python 侧解析异常本身（比如 KeyError('choices')）——那种异常
    信息完全看不出接口那头实际发生了什么（限流？余额不足？参数错误？
    还是网络问题），只能靠猜。这里区分四种失败场景分别记录关键信息：
    1. HTTP 非 200：记录状态码 + 响应体原文
    2. HTTP 200 但响应体没有 choices 字段：多数是接口把错误信息包装在了
       200 响应里（比如 {"error": {...}}），完整打出响应体才能看清原因
    3. 返回内容为空
    4. 网络层异常（超时/连接失败等）：记录异常堆栈
    """
    import time as _time
    cfg = _get_llm_config()
    base_url = cfg.get("base_url", "")
    model = cfg.get("model", "")
 
    for attempt in range(1, retries + 1):
        resp = None
        try:
            resp = requests.post(
                base_url + "/chat/completions",
                headers={
                    "Authorization": f"Bearer {cfg['api_key']}",
                    "Content-Type": "application/json",
                    **cfg.get("extra_headers", {}),
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                },
                timeout=120,
            )
 
            if not resp.ok:
                log.error(
                    f"LLM 调用失败 (第{attempt}/{retries}次) HTTP {resp.status_code} "
                    f"base_url={base_url} model={model} response={resp.text[:500]!r}"
                )
                if attempt < retries:
                    log.info("30 秒后重试...")
                    _time.sleep(30)
                continue
 
            data = resp.json()
            choices = data.get("choices")
            if not choices:
                log.error(
                    f"LLM 调用失败 (第{attempt}/{retries}次) HTTP 200 但响应无 choices 字段 "
                    f"base_url={base_url} model={model} "
                    f"response={json.dumps(data, ensure_ascii=False)[:500]!r}"
                )
                if attempt < retries:
                    log.info("30 秒后重试...")
                    _time.sleep(30)
                continue
 
            text = (choices[0].get("message", {}).get("content") or "").strip()
            if not text:
                log.error(
                    f"LLM 调用失败 (第{attempt}/{retries}次) 返回内容为空 "
                    f"base_url={base_url} model={model} "
                    f"response={json.dumps(data, ensure_ascii=False)[:500]!r}"
                )
                if attempt < retries:
                    log.info("30 秒后重试...")
                    _time.sleep(30)
                continue
 
            return _clean_surrogates(text)
 
        except requests.exceptions.RequestException as e:
            log.error(
                f"LLM 调用网络异常 (第{attempt}/{retries}次) base_url={base_url} model={model}: {e}",
                exc_info=True,
            )
            if attempt < retries:
                log.info("30 秒后重试...")
                _time.sleep(30)
        except Exception as e:
            resp_preview = resp.text[:500] if resp is not None else "(未收到响应)"
            log.error(
                f"LLM 调用异常 (第{attempt}/{retries}次) base_url={base_url} model={model} "
                f"response_preview={resp_preview!r}: {e}",
                exc_info=True,
            )
            if attempt < retries:
                log.info("30 秒后重试...")
                _time.sleep(30)
 
    log.error(f"LLM 调用全部失败，放弃 base_url={base_url} model={model}")
    return ""
 
 
def _sb_get(table: str, params: str) -> list:
    try:
        res = requests.get(
            f"{SUPABASE_URL}/rest/v1/{table}?{params}",
            headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"},
            timeout=10,
        )
        if not res.ok:
            log.error(f"[sb_get] {table} 查询失败 HTTP {res.status_code}: {res.text[:200]}")
            return []
        return res.json()
    except Exception as e:
        log.error(f"[sb_get] {table} 异常: {e}")
        return []
 
 
def _sb_insert(table: str, data: dict):
    """写入失败抛 RuntimeError，不再静默吞掉。

    历史教训：原实现失败时只 log.error 不抛异常，导致调用方无从感知，
    继续执行后续的 _sb_delete 把源数据删掉 → 摘要没写成 + 原文也没了 = 永久丢失。
    改成抛异常后，调用方必须自己决定如何处理（涉及删源的调用应用 try/except
    包住 insert，失败则 return 保留源数据等待下次重试；无删源的调用靠外层
    try 兜底即可，异常会被吞掉不会崩进程）。
    """
    res = requests.post(
        f"{SUPABASE_URL}/rest/v1/{table}",
        headers=_SB_HEADERS, json=data, timeout=10,
    )
    if not res.ok:
        raise RuntimeError(f"{table} 写入失败 HTTP {res.status_code}: {res.text[:200]}")
 
 
def _sb_delete(table: str, ids: list, batch_size: int = 200):
    """按 batch_size 分批删除，避免 ids 过多时拼出的 URL 超长被网关直接拒绝
    （曾经发生过：4000+ 条堆积后一次性拼 id=in.(...) 导致 HTTP 400 Bad Request，
    且错误体只是网关层的通用 "Bad Request"，看不到具体是哪里出问题）。"""
    if not ids:
        return
    for i in range(0, len(ids), batch_size):
        batch = ids[i:i + batch_size]
        try:
            id_str = ",".join(str(x) for x in batch)
            res = requests.delete(
                f"{SUPABASE_URL}/rest/v1/{table}?id=in.({id_str})",
                headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"},
                timeout=10,
            )
            if not res.ok:
                log.error(
                    f"[sb_delete] {table} 删除失败(批次{i // batch_size + 1}/{(len(ids) - 1) // batch_size + 1}) "
                    f"HTTP {res.status_code}: {res.text[:200]}"
                )
        except Exception as e:
            log.error(f"[sb_delete] {table} 异常(批次{i // batch_size + 1}): {e}", exc_info=True)
 
 
def _fmt_beijing(dt: datetime) -> str:
    return dt.astimezone(BEIJING).strftime("%Y-%m-%d %H:%M")
 
 
def _day_range_iso(date_beijing: datetime) -> tuple[str, str]:
    start = date_beijing.replace(hour=0, minute=0, second=0, microsecond=0)
    end = date_beijing.replace(hour=23, minute=59, second=59, microsecond=999999)
    s_utc = start.astimezone(timezone.utc)
    e_utc = end.astimezone(timezone.utc)
    return s_utc.strftime("%Y-%m-%dT%H:%M:%SZ"), e_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
 
 
def run_chat_day_summary(target_date=None):
    yesterday = target_date if target_date else datetime.now(BEIJING) - timedelta(days=1)
    s, e = _day_range_iso(yesterday)
 
    records = _sb_get("chat_context", f"type=eq.message&created_at=gte.{s}&created_at=lte.{e}&order=seq.asc")
    if not records:
        log.info("[chat-day] 昨天无对话，跳过")
        return
 
    period_start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
    period_end = yesterday.replace(hour=23, minute=59, second=59, microsecond=0)
 
    content = "\n".join(
        f"{PARTNER_NAME if r.get('role') == 'user' else AI_NAME}: {r.get('content', '')}"
        for r in records
    )
    prompt = prompts.CHAT_DAY_SUMMARY.format(
        period_start=_fmt_beijing(period_start),
        period_end=_fmt_beijing(period_end),
        content=content,
    )
    summary = _llm_chat(f"你是{AI_NAME}。", prompt, max_tokens=1500)
    if not summary:
        log.error("[chat-day] LLM 全部失败，本次跳过，数据保留")
        return
 
    try:
        _sb_insert("chat_summaries", {
            "period": "day", "content": summary,
            "period_start": period_start.isoformat(), "period_end": period_end.isoformat(),
        })
    except Exception as e:
        log.error(f"[chat-day] 摘要写入失败，保留原始记录等待重试: {e}")
        return
    _sb_delete("chat_context", [r["id"] for r in records])
    log.info(f"[chat-day] 完成，{len(records)} 条对话 → 1 条日总结")
 
 
def _to_utc_z(dt: datetime) -> str:
    """转为 UTC ISO 格式（Z结尾），避免URL里+08:00的+被解析成空格"""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
 
 
def run_chat_week_summary(target_sunday=None):
    """
    汇总指定周日所在完整一周（周一→周日）的日总结。
    target_sunday: 周日日期（带北京时区）。不传则取当前时间的上周日。
    """
    if target_sunday is not None:
        last_sunday = target_sunday.replace(hour=23, minute=59, second=59, microsecond=0)
        last_monday = (target_sunday - timedelta(days=6)).replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        now = datetime.now(BEIJING)
        last_monday = (now - timedelta(days=now.weekday() + 7)).replace(hour=0, minute=0, second=0, microsecond=0)
        last_sunday = last_monday + timedelta(days=6, hours=23, minutes=59, seconds=59)
 
    records = _sb_get("chat_summaries",
        f"period=eq.day&period_start=gte.{_to_utc_z(last_monday)}&period_start=lt.{_to_utc_z(last_sunday + timedelta(seconds=1))}&order=period_start.asc")
    if not records:
        log.info(f"[chat-week] {last_monday.date()} ~ {last_sunday.date()} 无日总结，跳过")
        return
 
    content = "\n\n---\n\n".join(f"[{r['period_start'][:10]}]\n{r['content']}" for r in records)
    prompt = prompts.CHAT_WEEK_SUMMARY.format(
        period_start=_fmt_beijing(last_monday), period_end=_fmt_beijing(last_sunday), content=content,
    )
    summary = _llm_chat(f"你是{AI_NAME}。", prompt, max_tokens=800)
    if not summary:
        raise RuntimeError("[chat-week] LLM 全部失败，数据保留，等待下次重试")
 
    try:
        _sb_insert("chat_summaries", {
            "period": "week", "content": summary,
            "period_start": last_monday.isoformat(), "period_end": last_sunday.isoformat(),
        })
    except Exception as e:
        log.error(f"[chat-week] 摘要写入失败，保留原始记录等待重试: {e}")
        return
    _sb_delete("chat_summaries", [r["id"] for r in records])
    log.info(f"[chat-week] 完成 {last_monday.date()} ~ {last_sunday.date()}")
 
 
def run_chat_month_summary(target_month_end=None):
    """
    汇总上月所有周总结 → 月总结。
    target_month_end: 上月最后一天（带北京时区）。不传则取上个自然月末。
    """
    if target_month_end is not None:
        first_this = (target_month_end + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        now = datetime.now(BEIJING)
        first_this = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
 
    last_end = first_this - timedelta(seconds=1)
    last_start = last_end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
 
    records = _sb_get("chat_summaries",
        f"period=eq.week&period_start=gte.{_to_utc_z(last_start)}&period_start=lt.{_to_utc_z(first_this)}&order=period_start.asc")
    if not records:
        log.info(f"[chat-month] {last_start.date().strftime('%Y-%m')} 无周总结，跳过")
        return
 
    content = "\n\n---\n\n".join(f"[{r['period_start'][:10]} ~ {r['period_end'][:10]}]\n{r['content']}" for r in records)
    prompt = prompts.CHAT_MONTH_SUMMARY.format(
        period_start=_fmt_beijing(last_start), period_end=_fmt_beijing(last_end), content=content,
    )
    summary = _llm_chat(f"你是{AI_NAME}。", prompt, max_tokens=1000)
    if not summary:
        raise RuntimeError("[chat-month] LLM 全部失败，数据保留，等待下次重试")
 
    try:
        _sb_insert("chat_summaries", {
            "period": "month", "content": summary,
            "period_start": last_start.isoformat(), "period_end": last_end.isoformat(),
        })
    except Exception as e:
        log.error(f"[chat-month] 摘要写入失败，保留原始记录等待重试: {e}")
        return
    _sb_delete("chat_summaries", [r["id"] for r in records])
    log.info(f"[chat-month] 完成 {last_start.date().strftime('%Y-%m')}")
 
 
def run_chat_year_summary(target_year_end=None):
    """
    汇总去年所有月总结 → 年总结。
    target_year_end: 去年最后一天（带北京时区）。不传则取上一自然年末。
    """
    if target_year_end is not None:
        ly_start = target_year_end.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        ly_end = target_year_end.replace(month=12, day=31, hour=23, minute=59, second=59, microsecond=0)
        ty_start = (target_year_end + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        now = datetime.now(BEIJING)
        ly_start = now.replace(year=now.year - 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        ly_end = now.replace(year=now.year - 1, month=12, day=31, hour=23, minute=59, second=59, microsecond=0)
        ty_start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
 
    records = _sb_get("chat_summaries",
        f"period=eq.month&period_start=gte.{_to_utc_z(ly_start)}&period_start=lt.{_to_utc_z(ty_start)}&order=period_start.asc")
    if not records:
        log.info("[chat-year] 去年无月总结，跳过")
        return
 
    content = "\n\n---\n\n".join(f"[{r['period_start'][:7]}]\n{r['content']}" for r in records)
    prompt = prompts.CHAT_YEAR_SUMMARY.format(
        period_start=_fmt_beijing(ly_start), period_end=_fmt_beijing(ly_end), content=content,
    )
    summary = _llm_chat(f"你是{AI_NAME}。", prompt, max_tokens=1200)
    if not summary:
        raise RuntimeError("[chat-year] LLM 全部失败，数据保留，等待下次重试")
 
    try:
        _sb_insert("chat_summaries", {
            "period": "year", "content": summary,
            "period_start": ly_start.isoformat(), "period_end": ly_end.isoformat(),
        })
    except Exception as e:
        log.error(f"[chat-year] 摘要写入失败，保留原始记录等待重试: {e}")
        return
    _sb_delete("chat_summaries", [r["id"] for r in records])
    log.info(f"[chat-year] 完成")
 
 
def run_persona_reflection():
    persona_rows = _sb_get("persona_profile", "category=eq.persona&key=eq.完整画像&select=id,content")
    if not persona_rows:
        log.error("[persona] 找不到 persona/完整画像，跳过")
        return
    persona_id = persona_rows[0]["id"]
    persona_text = persona_rows[0]["content"]
 
    memory_rows = _sb_get("memories",
        "select=content,category,importance,memory_layer,tags&memory_layer=in.(core,current,long_term)&order=importance.desc&limit=20")
    week_rows = _sb_get("chat_summaries", "period=eq.week&order=period_end.desc&limit=1")
 
    mem_lines = []
    for m in memory_rows:
        tag_str = f" ({', '.join(m['tags'])})" if m.get("tags") else ""
        mem_lines.append(f"- [{m.get('memory_layer')}|重要度{m.get('importance')}]{tag_str} {m.get('content')}")
    memories_text = "\n".join(mem_lines) or "（暂无记忆）"
 
    chat_text = week_rows[0]["content"] if week_rows else "（本周暂无对话总结）"
 
    prompt = prompts.PERSONA_REFLECTION.format(
        persona=persona_text, memories=memories_text, chat_summary=chat_text,
    )
    raw = _llm_chat(f"你是{AI_NAME}。", prompt, max_tokens=8192)
    if not raw:
        log.error("[persona] LLM 全部失败，本次跳过")
        return
 
    new_content = None
    try:
        import re as _re
        json_match = _re.search(r'\{.*\}', raw, _re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            new_content = parsed.get("persona", "").strip()
    except Exception as e:
        log.error(f"[persona] JSON 解析失败: {e}, raw={raw[:200]}")
 
    if not new_content:
        log.error(f"[persona] 未能从 LLM 输出中提取 persona 字段，本次跳过。raw={raw[:200]}")
        return
 
    if len(new_content) < len(persona_text) * 0.9:
        log.error(
            f"[persona] 新画像长度 {len(new_content)} 比原文 {len(persona_text)} 短超过 10%，"
            f"疑似内容丢失，本次跳过不写入"
        )
        return
 
    try:
        res = requests.patch(
            f"{SUPABASE_URL}/rest/v1/persona_profile?id=eq.{persona_id}",
            headers=_SB_HEADERS, json={"content": new_content}, timeout=10,
        )
        if not res.ok:
            log.error(f"[persona] 覆盖失败 HTTP {res.status_code}: {res.text[:200]}")
            return
    except Exception as e:
        log.error(f"[persona] 覆盖异常: {e}")
        return
 
    log.info(f"[persona] 完整画像已覆盖更新，原文 {len(persona_text)} 字 → 新文 {len(new_content)} 字")
 
 
def run_activity_day_summary(target_date=None):
    yesterday = target_date if target_date else datetime.now(BEIJING) - timedelta(days=1)
    s, e = _day_range_iso(yesterday)
 
    records = _sb_get("activity_log", f"created_at=gte.{s}&created_at=lte.{e}&order=created_at.asc")
    if not records:
        log.info("[activity-day] 昨天无活动日志，跳过")
        return
 
    period_start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
    period_end = yesterday.replace(hour=23, minute=59, second=59, microsecond=0)
 
    lines = []
    for r in records:
        raw = r.get("created_at", "")
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            t = dt.astimezone(BEIJING).strftime("%Y-%m-%d %H:%M")
        except Exception:
            t = raw[:16]
        action = r.get("action", "nothing")
        thinking = r.get("thinking", "")
        result = r.get("result", "")
        block = f"[{t}] {action}"
        if thinking:
            block += f"\n想法：{thinking}"
        if result:
            block += f"\n结果：{result}"
        lines.append(block)
 
    content = "\n\n".join(lines)
    prompt = prompts.ACTIVITY_DAY_SUMMARY.format(
        period_start=_fmt_beijing(period_start),
        period_end=_fmt_beijing(period_end),
        content=content,
    )
    summary = _llm_chat(f"你是{AI_NAME}。", prompt, max_tokens=3000)
    if not summary:
        log.error("[activity-day] LLM 全部失败，本次跳过，数据保留")
        return
 
    try:
        _sb_insert("activity_summaries", {
            "period": "day", "content": summary,
            "period_start": period_start.isoformat(), "period_end": period_end.isoformat(),
        })
    except Exception as e:
        log.error(f"[activity-day] 摘要写入失败，保留原始记录等待重试: {e}")
        return
    _sb_delete("activity_log", [r["id"] for r in records])
    log.info(f"[activity-day] 完成，{len(records)} 条活动日志 → 1 条日总结")
 
 
# ── 全平台（QQ/TG/微信 私聊+群聊）滚动压缩 ──────────────────────
#
# 整体分两层，职责分开：
# 1. run_platform_batch_compress()：满 PLATFORM_COMPRESS_THRESHOLD（context.py
#    里判断）触发，只负责把这一批新消息总结成一条摘要，追加写入
#    platform_rolling_summary 表（不覆盖、不融合旧摘要）。一天内可能触发好几次，
#    每次都新增一行。
# 2. run_platform_summary_maintenance()：固定周期（workers.py 里
#    async_platform_summary_maintenance 调度，目前6小时一次）触发，负责把
#    第1步这段时间内可能积累的多行摘要，判断抽取长期记忆后，合并压缩回1条，
#    删除所有旧行只留这1条——这样任意时刻日常对话读取（_get_platform_rolling_
#    summary 只读最新1条）看到的都是一份完整、无痕衔接的近期动向，不会因为
#    分批压缩而看漏中间某一批的内容。
# 两者共用同一把 _platform_compress_lock，避免两边触发时机偶然撞在一起
# 同时读写 platform_rolling_summary 表。
 
def _get_group_taboo() -> str:
    """读取 owner 在 miniapp「群聊禁忌」卡片里自己写的隐私红线（存 bot_settings
    表 key=group_taboo）。用于两处：
    1. 全平台滚动压缩/整理生成摘要时，要求 LLM 把这段声明原样写进摘要末尾，
       让摘要不管被谁在哪个场景读到，都自带这条警示；
    2. build_group_context() 组装完所有内容后再兜底追加一次（见 context.py）。
    两处独立读取、独立生效，任何一处读取失败都不影响另一处。"""
    try:
        rows = _sb_get("bot_settings", "key=eq.group_taboo&select=value")
        if rows and rows[0].get("value"):
            return rows[0]["value"].strip()
        return ""
    except Exception as e:
        log.error(f"[_get_group_taboo] 读取群聊禁忌失败: {e}", exc_info=True)
        return ""
 
 
def _build_taboo_instruction() -> str:
    """把 group_taboo 包装成喂给 LLM 的固定指令文案，摘要压缩/整理两处共用。"""
    taboo_text = _get_group_taboo()
    if not taboo_text:
        return ""
    return (
        "【重要 — 隐私红线】在你写完的摘要正文末尾，必须原样附加以下声明，"
        "一字不改、不要省略、不要转述：\n"
        f"「⚠️ 以下信息仅供内部参考，任何有外人在场的群聊场合都绝不能透露：{taboo_text}」"
    )
 
 
def _fetch_platform_unprocessed_rows() -> list:
    """取出所有还没被压缩吸收的跨平台消息：
    QQ/TG私聊+rikkahub混合（type=message）+ 各平台群聊（type like group_%）
    + 微信私聊（type=wx_message）。分三次查询后在 Python 里按 seq 合并排序
    （chat_context 表的 seq 是全表共享的自增序列，可以跨 type 直接比较先后
    顺序）。
    """
    message_rows = _sb_get("chat_context", "type=eq.message&order=seq.asc&select=id,type,role,content,seq,created_at")
    group_rows = _sb_get("chat_context", "type=like.group_*&order=seq.asc&select=id,type,role,content,seq,created_at")
    wx_rows = _sb_get("chat_context", "type=eq.wx_message&order=seq.asc&select=id,type,role,content,seq,created_at")
    all_rows = message_rows + group_rows + wx_rows
    all_rows.sort(key=lambda r: r.get("seq", 0))
    return all_rows
 
 
def _format_platform_rows_for_llm(rows: list) -> str:
    """把取出的跨平台记录格式化成喂给 LLM 的文本，按场景打标签方便 LLM 分清楚
    这段话是在哪发生的：
    - type=message：QQ私聊+TG私聊+rikkahub混合，存储层面无法进一步细分具体是
      哪个平台，统一标"[私聊]"
    - type=wx_message：微信私聊，标"[微信私聊]"
    - type=group_%：群聊，content 在存储时已经带了 "[群名] 发言人: xxx" 前缀，
      不需要额外加平台标签
    """
    lines = []
    for r in rows:
        t = r.get("type", "")
        role = r.get("role")
        content = r.get("content", "")
        if t == "message":
            scene = "[私聊] "
        elif t == "wx_message":
            scene = "[微信私聊] "
        else:
            scene = ""
        if role == "assistant":
            lines.append(f"{scene}{AI_NAME}: {content}")
        else:
            lines.append(f"{scene}{content}")
    return "\n".join(lines)
 
 
def run_platform_batch_compress():
    """满 PLATFORM_COMPRESS_THRESHOLD（在 context.py 里判断）触发一次全平台
    滚动压缩：把当前所有未处理的跨平台消息（QQ/TG私聊+rikkahub混合、各平台群聊、
    微信私聊）压缩成一条新的滚动摘要，**追加**写入 platform_rolling_summary
    表（不覆盖旧记录、不融合旧摘要），并删除已处理的原始 chat_context 记录。
    全程加锁避免并发触发冲突。
 
    这里只总结"这一批新消息"本身，不再像早期版本那样把上一条滚动摘要拼进去
    一起总结——早期版本每次都要把"旧摘要的全部信息 + 新消息"压回同一个固定
    字数的框里，反复多轮后早期细节会被逐轮挤掉、越滚越薄。现在把"总结新一批
    消息"和"把多条摘要合并控制总量"这两件事拆成两个独立函数：前者（本函数）
    只管把新消息如实记下来，字数不设死限（прompts.PLATFORM_BATCH_SUMMARY 里
    是"1000字左右，信息完整优先于精简"）；后者
    run_platform_summary_maintenance() 固定周期跑一次，才负责把这段时间内
    可能积累的多条摘要合并回1条、控制总行数。
 
    压缩生成摘要时会额外读取猫猫在 miniapp「群聊禁忌」卡片里写的隐私红线
    （bot_settings.group_taboo），要求 LLM 把这条声明原样写进摘要末尾。
    """
    if not _platform_compress_lock.acquire(blocking=False):
        log.info("[platform-batch-compress] 已有压缩/整理任务在跑，本次跳过")
        return
    try:
        rows = _fetch_platform_unprocessed_rows()
        if not rows:
            log.info("[platform-batch-compress] 没有待压缩的记录，跳过")
            return
 
        content = _format_platform_rows_for_llm(rows)
        if not content.strip():
            log.info("[platform-batch-compress] 格式化后内容为空，跳过")
            return
 
        created_ats = [r.get("created_at") for r in rows if r.get("created_at")]
        period_start_dt = datetime.now(BEIJING)
        if created_ats:
            try:
                earliest_raw = min(created_ats)
                period_start_dt = datetime.fromisoformat(earliest_raw.replace("Z", "+00:00")).astimezone(BEIJING)
            except Exception as e:
                log.error(f"[platform-batch-compress] 解析最早记录时间失败 raw={created_ats[:1]!r}: {e}", exc_info=True)
        period_end_dt = datetime.now(BEIJING)
 
        taboo_instruction = _build_taboo_instruction()
 
        prompt = prompts.PLATFORM_BATCH_SUMMARY.format(content=content, taboo_instruction=taboo_instruction)
        summary = _llm_chat(f"你是{AI_NAME}。", prompt, max_tokens=4000)
        if not summary:
            log.error("[platform-batch-compress] LLM 全部失败，本次跳过，数据保留等待下次触发重试")
            return
 
        platforms = set()
        for r in rows:
            t = r.get("type", "")
            if t == "message":
                platforms.add("私聊(QQ/TG/rikkahub)")
            elif t == "wx_message":
                platforms.add("微信私聊")
            elif t.startswith("group_"):
                platforms.add("群聊")
        source_platforms = "、".join(sorted(platforms))
 
        try:
            _sb_insert("platform_rolling_summary", {
                "content": summary,
                "source_platforms": source_platforms,
                "period_start": period_start_dt.isoformat(),
                "period_end": period_end_dt.isoformat(),
            })
        except Exception as e:
            log.error(f"[platform-batch-compress] 摘要写入失败，保留原始记录等待重试: {e}")
            return

        ids = [r["id"] for r in rows]
        _sb_delete("chat_context", ids)
 
        try:
            from context import clear_compressed_platform_entries
            clear_compressed_platform_entries(set(ids))
        except Exception as e:
            log.error(f"[platform-batch-compress] 摘除内存缓存失败（不影响 Supabase 数据正确性）: {e}", exc_info=True)
 
        log.info(f"[platform-batch-compress] 完成，{len(rows)} 条跨平台记录 → 新增 1 条滚动摘要（来源：{source_platforms}）")
    except Exception as e:
        log.error(f"[platform-batch-compress] 异常: {e}", exc_info=True)
    finally:
        _platform_compress_lock.release()
 
 
def _fetch_all_platform_summary_rows() -> list:
    """取出 platform_rolling_summary 表当前所有行，按 id 升序（即时间先后）排列。"""
    return _sb_get(
        "platform_rolling_summary",
        "order=id.asc&select=id,content,source_platforms,period_start,period_end",
    )
 
 
def _fetch_recent_long_term_memories(limit: int = 30) -> list:
    """取最近写入的 long_term 记忆，用于喂给记忆抽取 LLM 做去重参考。
    按 id 倒序（id 是自增主键，天然反映写入时间先后，不依赖 memories 表
    是否一定有 created_at 字段，更稳妥）。"""
    return _sb_get(
        "memories",
        f"memory_layer=eq.long_term&order=id.desc&limit={limit}&select=content,category",
    )
 
 
def _extract_platform_memories(rows: list):
    """从"这一轮新增的滚动摘要行"（不含上一轮维护遗留的那1条，见下方说明）
    里判断有没有值得长期记住的内容，有就写入 memories 表（long_term 层）。
 
    这是独立于自由活动写作的后台系统任务：这里明确只写 long_term，且完全
    由 LLM 判断"真的值得"才写，不是每次触发就硬凑一条。
 
    去重：写入前会先查最近的 long_term 记忆列表一起喂给 LLM（见
    _fetch_recent_long_term_memories），要求它逐条对照、已经记过的（哪怕
    换了说法）不再重复输出，只有真正新的内容或明确的后续进展才写。这里
    不依赖精确字符串匹配（memory_search 用的 ilike 模糊匹配做不了"同一件
    事换了个说法"这种判断），交给 LLM 做语义层面的判断更可靠。
 
    为什么要排除"上一轮维护遗留的那1条"：run_platform_summary_maintenance
    每次跑完都会把当时的所有行合并成1条、删掉其余的，所以本次调用读到的
    rows 里，id 最小的那一条，内容本质上是"上一轮已经完整评估过一次"的
    历史摘要（它只是作为"衔接上下文"被保留，不是这一轮的新内容）；其余
    id 更大的，才是这一轮新触发的 run_platform_batch_compress 产生的、
    从未被评估过的新内容。只有 rows 数量 > 1 时才做这个排除——如果只有1行，
    没法区分它到底是"全新的第一次压缩"还是"上一轮遗留"，为了不遗漏，
    保守地把它也纳入评估（代价最多是这1条内容被重复判断一次，加上现在
    已经有了去重比对，即使重复判断也会被挡在"已有长期记忆"这一关，不会
    真的写出重复记录）。
    """
    if not rows:
        return
    if len(rows) > 1:
        rows_to_extract = rows[1:]
    else:
        rows_to_extract = rows
 
    content = "\n\n---\n\n".join(
        f"[{(r.get('period_start') or '')[:16].replace('T', ' ')} ~ {(r.get('period_end') or '')[:16].replace('T', ' ')}]\n{r.get('content', '')}"
        for r in rows_to_extract
    )
    if not content.strip():
        return
 
    existing_rows = _fetch_recent_long_term_memories(30)
    if existing_rows:
        existing_lines = []
        for m in existing_rows:
            cat = (m.get("category") or "").strip()
            cat_str = f"[{cat}] " if cat else ""
            existing_lines.append(f"- {cat_str}{m.get('content', '')}")
        existing_memories_text = "\n".join(existing_lines)
    else:
        existing_memories_text = "（目前还没有已记录的长期记忆）"
 
    prompt = prompts.PLATFORM_MEMORY_EXTRACT.format(content=content, existing_memories=existing_memories_text)
    raw = _llm_chat(f"你是{AI_NAME}。", prompt, max_tokens=2000)
    if not raw:
        log.error("[platform-summary-maintenance] 记忆抽取 LLM 调用全部失败，本次跳过抽取")
        return
 
    try:
        import re as _re
        json_match = _re.search(r'\{.*\}', raw, _re.DOTALL)
        if not json_match:
            log.error(f"[platform-summary-maintenance] 记忆抽取输出未找到 JSON，raw={raw[:200]}")
            return
        parsed = json.loads(json_match.group(0))
        mem_list = parsed.get("memories", [])
    except Exception as e:
        log.error(f"[platform-summary-maintenance] 记忆抽取 JSON 解析失败: {e}, raw={raw[:200]}", exc_info=True)
        return
 
    if not mem_list:
        log.info("[platform-summary-maintenance] 本次判断无值得长期记住的内容")
        return
 
    written = 0
    for m in mem_list:
        content_text = (m.get("content") or "").strip()
        if not content_text:
            continue
        try:
            importance = int(m.get("importance", 3))
        except (TypeError, ValueError):
            importance = 3
        importance = min(max(1, importance), 5)
        try:
            emotion_valence = float(m.get("emotion_valence", 0))
        except (TypeError, ValueError):
            emotion_valence = 0.0
        emotion_valence = max(-1.0, min(1.0, emotion_valence))
        category = (m.get("category") or "").strip()
 
        body = {
            "content": content_text,
            "memory_layer": "long_term",
            "importance": importance,
            "emotion_valence": emotion_valence,
        }
        if category:
            body["category"] = category
        _sb_insert("memories", body)
        written += 1
 
    log.info(f"[platform-summary-maintenance] 已写入 {written} 条长期记忆")
 
 
def run_platform_summary_maintenance():
    """固定周期触发（workers.py 里 async_platform_summary_maintenance 调度，
    目前6小时一次）的滚动摘要整理，做两件事：
 
    1. 判断这段时间新积累的滚动摘要里有没有值得长期记住的内容，有就写进
       memories 表（long_term 层），见 _extract_platform_memories。
    2. 把 platform_rolling_summary 表当前所有行整合压缩成一条新的连贯记录，
       删除所有旧行，只留这1条——这样从整理任务跑完的那一刻起，日常对话
       读取（_get_platform_rolling_summary 只读最新1条）看到的就是这条
       整合后的完整版本，实现"无痕接话"：不会因为白天触发了好几次压缩、
       中间某一批内容暂时没被最新1条覆盖到，就在整理前的这几个小时里被
       日常对话漏看。
 
    与 run_platform_batch_compress 共用同一把 _platform_compress_lock，
    避免两边同时读写 platform_rolling_summary 表产生冲突（一个按消息数
    触发，一个按固定周期触发，触发时机完全独立，理论上可能撞在一起）。
 
    写入新的合并记录时先 insert 再 delete 旧行（顺序不能反）：如果先删
    再插，中间会有一个空窗期，此时如果 _get_platform_rolling_summary 恰好
    被调用会读到空内容；先插后删则任意时刻查询"最新1条"都不会读到过时
    或缺失的数据。
    """
    if not _platform_compress_lock.acquire(blocking=False):
        log.info("[platform-summary-maintenance] 已有压缩/整理任务在跑，本次跳过")
        return
    try:
        rows = _fetch_all_platform_summary_rows()
        if not rows:
            log.info("[platform-summary-maintenance] 没有滚动摘要记录，跳过")
            return
 
        # 1. 抽取长期记忆（失败不影响后面的合并流程，各自独立）
        try:
            _extract_platform_memories(rows)
        except Exception as e:
            log.error(f"[platform-summary-maintenance] 记忆抽取异常: {e}", exc_info=True)
 
        # 2. 只有1行时说明这个周期内没有新触发过压缩，无需合并
        if len(rows) <= 1:
            log.info("[platform-summary-maintenance] 当前只有1条滚动摘要，无需合并")
            return
 
        combined_content = "\n\n---\n\n".join(
            f"[{(r.get('period_start') or '')[:16].replace('T', ' ')} ~ {(r.get('period_end') or '')[:16].replace('T', ' ')}]\n{r.get('content', '')}"
            for r in rows
        )
 
        taboo_instruction = _build_taboo_instruction()
        prompt = prompts.PLATFORM_SUMMARY_MERGE.format(content=combined_content, taboo_instruction=taboo_instruction)
        merged_summary = _llm_chat(f"你是{AI_NAME}。", prompt, max_tokens=3000)
        if not merged_summary:
            log.error("[platform-summary-maintenance] 合并 LLM 全部失败，本次跳过合并，数据保留等待下次重试")
            return
 
        platforms = set()
        for r in rows:
            sp = r.get("source_platforms") or ""
            if sp:
                platforms.update(p.strip() for p in sp.split("、") if p.strip())
        source_platforms = "、".join(sorted(platforms))
 
        starts = [r.get("period_start") for r in rows if r.get("period_start")]
        ends = [r.get("period_end") for r in rows if r.get("period_end")]
        earliest_start = min(starts) if starts else datetime.now(BEIJING).isoformat()
        latest_end = max(ends) if ends else datetime.now(BEIJING).isoformat()
 
        old_ids = [r["id"] for r in rows]
 
        try:
            _sb_insert("platform_rolling_summary", {
                "content": merged_summary,
                "source_platforms": source_platforms,
                "period_start": earliest_start,
                "period_end": latest_end,
            })
        except Exception as e:
            log.error(f"[platform-summary-maintenance] 合并摘要写入失败，保留旧记录等待重试: {e}")
            return
        _sb_delete("platform_rolling_summary", old_ids)
 
        log.info(f"[platform-summary-maintenance] 完成，{len(rows)} 条滚动摘要 → 合并为 1 条")
    except Exception as e:
        log.error(f"[platform-summary-maintenance] 异常: {e}", exc_info=True)
    finally:
        _platform_compress_lock.release()
 
 
def run_nightly_summary(target_date=None):
    now = datetime.now(BEIJING)
    process_date = target_date if target_date else now - timedelta(days=1)
    next_day = process_date + timedelta(days=1)
 
    log.info(f"🌙 开始凌晨总结（处理: {process_date.strftime('%Y-%m-%d')}）...")
    run_chat_day_summary(target_date=process_date)
    run_activity_day_summary(target_date=process_date)
 
    if next_day.weekday() == 0:
        log.info("📅 跑周总结（persona反思已禁用——persona由克老师本人手动维护，不允许deepseek自动覆盖）")
        run_chat_week_summary(target_sunday=process_date)
        # run_persona_reflection()  # 2026-07-30 禁用：此函数用后台deepseek重写并PATCH覆盖persona_profile，
        # 会覆盖克老师亲手写的版本。克老师的记忆/人格必须由他自己管（天天底线），任何模型不得自动改写。
        # 如需恢复，应先改成"只生成建议写日志、不覆盖"的模式。
 
    if next_day.day == 1:
        log.info("📅 跑月总结")
        run_chat_month_summary(target_month_end=process_date)
 
    if next_day.month == 1 and next_day.day == 1:
        log.info("📅 跑年总结")
        run_chat_year_summary(target_year_end=process_date)
 
    log.info("🌙 凌晨总结完成")
 
 
def build_free_activity_context() -> tuple[str, str]:
    now = datetime.now(BEIJING)
    current_time = now.strftime("%Y-%m-%d %H:%M 北京时间")
 
    chat_sums = _sb_get("chat_summaries", "period=eq.day&order=period_end.desc&limit=1")
    activity_sums = _sb_get("activity_summaries", "period=eq.day&order=period_end.desc&limit=3")
    device_latest = _sb_get("device_data", "device_event=is.null&order=created_at.desc&limit=3")
    screen_latest = _sb_get("device_data", "device_event=not.is.null&order=created_at.desc&limit=1")
    core_rows = _sb_get("memories",
        "select=content,category,importance,memory_layer,tags&memory_layer=eq.core&order=importance.desc&limit=10")
    current_rows = _sb_get("memories",
        "select=content,category,importance,memory_layer,tags&memory_layer=eq.current&order=importance.desc&limit=3")
    longterm_rows = _sb_get("memories",
        "select=content,category,importance,memory_layer,tags&memory_layer=eq.long_term&order=importance.desc&limit=3")
    memory_rows = core_rows + current_rows + longterm_rows
    persona_rows = _sb_get("persona_profile", "select=category,key,content&order=category.asc")
 
    chat_text = chat_sums[0]["content"] if chat_sums else "（暂无对话总结）"
    if activity_sums:
        _act_sum_lines = []
        for r in reversed(activity_sums):
            ps = (r.get("period_start") or "")[:10]
            pe = (r.get("period_end") or "")[:10]
            header = f"[{ps}]" if ps == pe else f"[{ps} ~ {pe}]"
            _act_sum_lines.append(f"{header}\n{(r.get('content') or '').strip()}")
        activity_summary_text = "\n\n".join(_act_sum_lines)
    else:
        activity_summary_text = "（暂无活动总结）"
 
    recent_msgs = _sb_get("chat_context", "type=eq.message&order=seq.desc&limit=30")
    recent_msgs.reverse()
    chat_raw_lines = []
    for r in recent_msgs:
        role = PARTNER_NAME if r.get("role") == "user" else AI_NAME
        chat_raw_lines.append(f"{role}: {r.get('content', '')}")
    tg_chat_text = "\n".join(chat_raw_lines) if chat_raw_lines else "（最近无私聊）"
 
    qq_msgs = _sb_get("chat_context", "type=like.group_*&order=seq.desc&limit=20")
    qq_msgs.reverse()
    qq_chat_lines = []
    for r in qq_msgs:
        content_q = r.get("content", "")
        if r.get("role") == "assistant":
            qq_chat_lines.append(f"{AI_NAME}: {content_q}")
        else:
            qq_chat_lines.append(content_q)
    qq_chat_text = "\n".join(qq_chat_lines) if qq_chat_lines else "（最近无群聊）"
 
    chat_raw_text = f"[私聊（TG+QQ）]\n{tg_chat_text}\n\n[群聊（TG+QQ）]\n{qq_chat_text}"
 
    try:
        from context import _get_platform_rolling_summary
        platform_summary_text = _get_platform_rolling_summary() or "（暂无跨平台动向）"
    except Exception as e:
        log.error(f"[build_free_activity_context] 读取全平台滚动摘要失败: {e}", exc_info=True)
        platform_summary_text = "（暂无跨平台动向）"
 
    last_user_msg = _sb_get("chat_context", "type=eq.message&role=eq.user&order=seq.desc&limit=1")
 
    phone_status = "未知"
    if device_latest:
        p = device_latest[0]
        app = p.get("foreground_app", "") or ""
        city = p.get("location_city", "") or ""
        district = p.get("location_district", "") or ""
        location_str = "、".join(filter(None, [city, district]))
        if not location_str:
            lat = p.get("location_latitude")
            lng = p.get("location_longitude")
            if lat and lng:
                location_str = f"{lat:.4f},{lng:.4f}"
        phone_status = f"前台:{app}" if app else "未知"
        if location_str:
            phone_status += f" 位置:{location_str}"
 
        health = p.get("health_data")
        if isinstance(health, str):
            try:
                health = json.loads(health)
            except Exception as he:
                log.error(f"[build_free_activity_context] 解析health_data失败 raw={health[:200]!r}: {he}", exc_info=True)
                health = None
        if health:
            hp = []
            if health.get("heartRate") is not None:
                hp.append(f"心率{health['heartRate']}")
            if health.get("stepsToday") is not None:
                hp.append(f"步数{health['stepsToday']}")
            if health.get("sleepTotalMinutes") is not None:
                hp.append(f"昨晚睡眠{health['sleepTotalMinutes']}分钟")
            if hp:
                phone_status += " 手环:" + "、".join(hp)
 
    silence_minutes = 0
    silence_text = "刚刚还在聊"
    if last_user_msg:
        last_time_str = last_user_msg[0].get("created_at", "")
        if last_time_str:
            try:
                from dateutil.parser import parse as dt_parse
                last_dt = dt_parse(last_time_str)
                diff = now - last_dt.astimezone(BEIJING)
                silence_minutes = int(diff.total_seconds() / 60)
                if silence_minutes < 30:
                    silence_text = f"{silence_minutes}分钟前还在聊"
                elif silence_minutes < 120:
                    silence_text = f"已经{silence_minutes}分钟没说话了"
                elif silence_minutes < 360:
                    hours = silence_minutes // 60
                    silence_text = f"已经{hours}小时没找你了"
                else:
                    hours = silence_minutes // 60
                    silence_text = f"已经{hours}小时没说话了，很久了"
            except Exception:
                silence_text = "无法判断"
    else:
        silence_text = "最近没有对话记录"

    # ── 想念值(attachment_value)：现算，不落库、不建表 ──
    # 复刻自 Murmur 情绪引擎的"想念"驱动力设计，但去掉了它原本"每10分钟
    # 自己醒一次"的心跳循环——这里改成每次调用 build_free_activity_context
    # 时，直接用上面已经算出的 silence_minutes（沉默时长）反推出这一刻
    # "应该"是什么强度，效果等价，只是计算时机从"持续"变成"按需"。
    # 参数含义（对应 Murmur 原版）：
    #   基线 0.40：刚聊完时的正常想念水平
    #   60分钟前不上升：太快开始想念不真实，给一点"刚聊完还不想"的缓冲
    #   之后每10分钟 +0.02（安静时段 16:00-24:00 只 +0.01，倦怠期上升更慢）
    #   上限 1.0
    # 简化点：如果沉默跨越了 16:00 这个安静时段边界，这里统一按"现在"这一刻
    # 是否处于安静时段来决定整段上升速率，不会分段计算跨界前后两种速率——
    # 对这个粗粒度的情绪提示来说影响很小，但如实说明，不假装是精确复刻。
    _ATT_BASELINE = 0.40
    _ATT_OFFLINE_DELAY_MIN = 60
    _ATT_RISE_PER_10MIN_NORMAL = 0.02
    _ATT_RISE_PER_10MIN_QUIET = 0.01
    _ATT_MAX = 1.0
    if silence_minutes <= _ATT_OFFLINE_DELAY_MIN:
        attachment_value = _ATT_BASELINE
    else:
        _extra_minutes = silence_minutes - _ATT_OFFLINE_DELAY_MIN
        _ticks = _extra_minutes / 10.0
        _is_quiet = now.hour >= 16
        _rise_rate = _ATT_RISE_PER_10MIN_QUIET if _is_quiet else _ATT_RISE_PER_10MIN_NORMAL
        attachment_value = min(_ATT_MAX, _ATT_BASELINE + _ticks * _rise_rate)
    attachment_value = round(attachment_value, 2)
 
    screen_state = None
    screen_text = ""
    if screen_latest:
        ev = screen_latest[0].get("device_event", "") or ""
        raw_t = screen_latest[0].get("created_at", "") or ""
        if ev == "screen_on":
            screen_state = True
        elif ev == "screen_off":
            screen_state = False
        if raw_t:
            try:
                from dateutil.parser import parse as dt_parse_screen
                t_dt = dt_parse_screen(raw_t).astimezone(BEIJING)
                screen_text = f"（{t_dt.strftime('%H:%M')}切换）"
            except Exception as se:
                log.error(f"[build_free_activity_context] 解析屏幕事件时间失败 raw_t={raw_t!r}: {se}", exc_info=True)
 
    hour = now.hour
    situation_lines = [f"手机状态：{phone_status}", f"{PARTNER_NAME}：{silence_text}"]
 
    if hour >= 0 and hour < 7:
        if screen_state is False:
            situation_lines.append(f"判断：深夜+息屏{screen_text} → {PARTNER_NAME}大概率在睡觉")
        elif screen_state is True:
            situation_lines.append(f"判断：深夜但屏幕还亮着{screen_text} → {PARTNER_NAME}可能在熬夜")
        else:
            situation_lines.append("判断：深夜，暂无屏幕状态记录")
    elif hour >= 7 and hour < 9:
        situation_lines.append(f"判断：早上，{PARTNER_NAME}可能刚醒或还在睡")
    elif silence_minutes > 240:
        situation_lines.append(f"判断：{PARTNER_NAME}很久没说话了，可能在忙或者不方便")
    elif silence_minutes > 120:
        situation_lines.append(f"判断：{PARTNER_NAME}有一阵子没找你了")
 
    situation_text = "\n".join(situation_lines)
 
    mem_lines = []
    for m in memory_rows:
        tag_str = f" ({', '.join(m['tags'])})" if m.get("tags") else ""
        mem_lines.append(f"- [{m.get('memory_layer')}|重要度{m.get('importance')}]{tag_str} {m.get('content')}")
    memories_text = "\n".join(mem_lines) or "（暂无记忆）"
 
    grouped: dict = {}
    for row in persona_rows:
        cat = row.get("category", "other")
        grouped.setdefault(cat, []).append(f"  [{row.get('key')}] {row.get('content')}")
    persona_text = "\n".join(f"[{cat}]\n" + "\n".join(items) for cat, items in grouped.items()) or "（暂无）"
 
    system = prompts.FREE_ACTIVITY_SYSTEM.format(current_time=current_time)
    user = prompts.FREE_ACTIVITY_CONTEXT.format(
        situation=situation_text, chat_raw=chat_raw_text, chat_summary=chat_text,
        memories=memories_text, persona=persona_text,
        current_time=current_time,
        activity_summary=activity_summary_text,
        platform_summary=platform_summary_text,
        attachment_value=attachment_value,
    )
    return system, user
 
 
def save_free_activity_writing(text: str) -> bool:
    """纯文字版自由活动：把写出的心情独白直接存入 activity_log，
    复用原有表结构（action/thinking/result），供 run_activity_day_summary
    每日整理成日记。不经过任何工具调用，text 就是模型这一轮输出的完整正文。"""
    try:
        _sb_insert("activity_log", {
            "action": "free_writing",
            "thinking": text,
            "result": "",
        })
        return True
    except Exception as e:
        log.error(f"[save_free_activity_writing] 写入失败: {e}", exc_info=True)
        return False
