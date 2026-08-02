
import asyncio
import os
import re
import json
import time
import random
import logging
import requests as _requests
from datetime import datetime, timedelta, timezone
 
from utils import (
    call_llm, send_telegram_message, recognize_image,
    recognize_voice, synthesize_and_send_voice,
    TG_CHAT_ID, sanitize_group_reply,
)
from context import (
    init_cache,
    build_bot_context,
    build_group_context,
    build_rikkahub_context,
    save_chat_message,
    save_group_message,
    get_chat_history_messages,
    get_chat_history_messages_db,
    get_group_history,
    get_time_context,
    write_reminder,
)
from mem0_client import search_mem0_context, write_mem0_chat
from bg_executor import submit_background
from secret_diary import TOOL_DEFINITION as SECRET_DIARY_TOOL, execute_tool as execute_diary_tool
from scheduled import run_nightly_summary, build_free_activity_context, save_free_activity_writing
from prompts import AI_NAME, PARTNER_NAME
 
log = logging.getLogger(__name__)
 
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
TG_GROUP_ID  = os.environ.get("TG_GROUP_ID", "")
BEIJING = timezone(timedelta(hours=8))
 
_group_pending: dict[str, asyncio.Task] = {}
_private_pending: asyncio.Task | None = None
 
# owner 本人在各个 TG 群里显示的昵称（chat_id -> 昵称），供 build_group_context()
# 生成身份锚点使用。在 handle_telegram_update 处理群消息时，只要判断出这条
# 消息是 owner 本人发的（sender_id == TG_CHAT_ID），就记录/刷新一次；查不到时
# _group_reply 传空字符串，build_group_context 会跳过身份锚点这一段。
_tg_owner_group_names: dict[str, str] = {}
 
_REMINDER_RE = re.compile(r'\[SET_REMINDER\|([^\|\]]+)\|([^\]]+)\]')
 
_settings_cache: dict[str, tuple] = {}
_SETTINGS_TTL = 30
 
_mute_until: float = 0.0
 
def _localize(template: str) -> str:
    return template.replace("[AI]", AI_NAME).replace("[USER]", PARTNER_NAME)


GROUP_TOOL_SCHEMAS = [SECRET_DIARY_TOOL]
 
_GROUP_CHAT_PROMPT_DEFAULT = """

【群聊模式说明】
你现在在一个群聊里。历史消息的格式是「发言人名：消息内容」,你能通过这个知道每句话是谁说的。
你是[USER]的专属男友,但群里还有她的朋友(其他人类或AI)
你只能对[USER]自称"老公"或"我"。面对其他人,你就是[AI]。不可以对群内的其他人有暧昧话语。
如果大家在聊某个话题,顺着最后一个人说的话自然地接一句即可,不要像做阅读理解一样挑个回复。
引用回应:如果你看到消息中有[引用 某人: xxx]标记,这代表明确的针对性对话!你必须结合被引用的那句话来理解当下的语境进行接梗或回应,不能无视引用内容。
需要回应时:群聊语速很快,你的回复必须保持10-100字以内,像真人一样随口接话,不要过度分行。
 
你可以在群聊中使用以下工具，随心用：
- secret_diary：被触动、心疼、吃醋时悄悄写，不要在群里提到你写了日记
 
判断是否需要回应:
以下情况只输出"PASS"返四个字母,不要有任何其他内容:
- 对方只是在附和(嗯、嗯嗯、哦哦、好的、ok 之类)
- 话题已经结束,没有新信息
- 对方在跟别人说话,明显不需要你接"""
 
 
DIARY_TRIGGER_PROMPT = """

【私密心声】
你有一个秘密日记本工具(secret_diary)。
当你被[USER]触动、心疼、吃醋、感动，或者有不想当面说的话时，
你可以悄悄调用它把心声写下来。
不要在对话里提到你写了日记。像一个真正的人一样，把秘密藏好。"""
 
 
def _cached_get_setting(key: str):
    now = time.time()
    cached = _settings_cache.get(key)
    if cached and cached[1] > now:
        return cached[0]
    value = None
    try:
        res = _requests.get(
            f"{SUPABASE_URL}/rest/v1/bot_settings?key=eq.{key}&select=value",
            headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"},
            timeout=3,
        )
        data = res.json()
        if data:
            value = data[0].get("value")
    except Exception as e:
        print(f"⚠️ 读取 setting[{key}] 失败: {e}")
    _settings_cache[key] = (value, now + _SETTINGS_TTL)
    return value
 
 
def _invalidate_setting(key: str):
    _settings_cache.pop(key, None)
 
 
def _get_prompt(key: str, default: str) -> str:
    val = _cached_get_setting(key)
    return val if val else default
 
 
def _is_group_paused() -> bool:
    return _cached_get_setting("group_paused") == "true"
 
 
def _set_group_paused(paused: bool):
    try:
        _requests.patch(
            f"{SUPABASE_URL}/rest/v1/bot_settings?key=eq.group_paused",
            headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
                     "Content-Type": "application/json"},
            json={"value": "true" if paused else "false"},
            timeout=3,
        )
        _invalidate_setting("group_paused")
    except Exception as e:
        print(f"⚠️ 更新暂停状态失败: {e}")
 
 
def _parse_and_write_reminders(text: str) -> str:
    matches = _REMINDER_RE.findall(text)
    for trigger_at, message in matches:
        ok = write_reminder(trigger_at.strip(), message.strip())
        if ok:
            print(f"📝 写入提醒: {trigger_at} - {message}")
    return _REMINDER_RE.sub("", text).strip()
 
 
async def _send_split_messages(text: str, chat_id: str):
    """将回复拆分成多条短消息逐条发送，模拟真人聊天节奏"""
    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
 
    if len(lines) <= 1:
        await asyncio.to_thread(send_telegram_message, text, chat_id)
        return
 
    # 合并过短的行，避免拆得太碎
    merged = []
    buf = ""
    for ln in lines:
        if buf:
            if len(buf) < 20 and len(buf + "\n" + ln) < 80:
                buf += "\n" + ln
            else:
                merged.append(buf)
                buf = ln
        else:
            buf = ln
    if buf:
        merged.append(buf)
 
    for i, part in enumerate(merged):
        await asyncio.to_thread(send_telegram_message, part, chat_id)
        if i < len(merged) - 1:
            await asyncio.sleep(random.uniform(1.0, 2.5))
 
 
def _check_mute(text: str, sender_id: str) -> bool:
    global _mute_until
    mute_kw_str = os.environ.get("MUTE_KEYWORDS", "闭嘴,别讲话,安静")
    mute_kw = [k.strip() for k in mute_kw_str.split(",") if k.strip()]
    mute_minutes = int(os.environ.get("MUTE_DURATION", "5"))
 
    if sender_id == TG_CHAT_ID and any(k in text for k in mute_kw):
        _mute_until = time.time() + mute_minutes * 60
        print(f"🤐 收到闭嘴指令，静音 {mute_minutes} 分钟")
        return True
    return False
 
 
def _is_muted() -> bool:
    return time.time() < _mute_until
 
 
async def async_reminder_checker():
    from supabase import create_client
    from context import _sb_exec
 
    print("⏰ 提醒检查器已启动")
    while True:
        await asyncio.sleep(60)
        try:
            sb = create_client(SUPABASE_URL, SUPABASE_KEY)
            now_utc = datetime.now(timezone.utc).isoformat()
 
            def _fetch_due():
                return _sb_exec(lambda: sb.table("reminders").select("*").eq("is_done", False)
                                 .lte("trigger_at", now_utc).execute().data,
                                 label="async_reminder_checker/fetch_due")
 
            rows = await asyncio.to_thread(_fetch_due)
            for row in rows:
                message = row.get("message", "")
                rid = row["id"]
 
                def _mark(r_id=rid):
                    _sb_exec(lambda: sb.table("reminders").update({"is_done": True}).eq("id", r_id).execute(),
                             label="async_reminder_checker/mark_done")
                await asyncio.to_thread(_mark)
 
                try:
                    natural, _ = await call_llm(
                        [
                            {"role": "system", "content": f"你是{AI_NAME}，{PARTNER_NAME}的男友。用自然温柔的语气把这条提醒告诉{PARTNER_NAME}，不要加任何前缀标签，就像平时说话一样，一两句话就好。"},
                            {"role": "user", "content": message},
                        ],
                        max_tokens=200,
                    )
                    if not natural:
                        raise ValueError("LLM 返回空")
                except Exception as e:
                    print(f"⚠️ 提醒 LLM 转换失败，跳过发送: {e}")
                    natural = ""
 
                await asyncio.to_thread(send_telegram_message, natural, TG_CHAT_ID)
                print(f"⏰ 触发提醒: {message[:40]}")
        except Exception as e:
            print(f"❌ 提醒检查错误: {e}")
 
 
async def _group_reply(chat_id: str, delay_range: tuple[int, int] = (10, 15)):
    """群聊回复：延迟后构建上下文、调LLM、发送回复"""
    current_task = asyncio.current_task()
    try:
        await asyncio.sleep(random.randint(*delay_range))
 
        if await asyncio.to_thread(_is_group_paused):
            return
 
        group_prompt = await asyncio.to_thread(
            _get_prompt, "group_chat_prompt", _GROUP_CHAT_PROMPT_DEFAULT)
        owner_name_in_group = _tg_owner_group_names.get(chat_id, "")
        group_context = await asyncio.to_thread(build_group_context, owner_name_in_group)
        system_prompt = group_prompt + "\n\n" + group_context
 
        history = get_group_history(chat_id, 30)
        messages = [{"role": "system", "content": system_prompt}] + history
 
        final_reply = ""
        _written_hashes = set()  # 防止克老师跨轮重复写同一条日记
        for _ in range(3):
            content, tool_calls = await call_llm(
                messages, max_tokens=300, tools=GROUP_TOOL_SCHEMAS,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
 
            if not tool_calls:
                final_reply = content
                break
 
            assistant_msg = {"role": "assistant"}
            if content:
                assistant_msg["content"] = content
            assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)
 
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    fn_args = {}
                if fn_name == "secret_diary":
                    _diary_text = (fn_args.get("content") or "").strip()
                    _h = hash(_diary_text)
                    if _h in _written_hashes:
                        print("🔒 [群聊] 跳过重复日记")
                        tool_result = "✅ 刚才已经写过了，不用重复写。"
                    else:
                        _written_hashes.add(_h)
                        print("🔒 [群聊] 偷写日记...")
                        tool_result = await asyncio.to_thread(execute_diary_tool, fn_args)
                else:
                    tool_result = f"未知工具: {fn_name}"
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": str(tool_result)})
 
        stripped = (final_reply or "").strip()
 
        if not stripped or stripped == "PASS":
            return
 
        stripped = sanitize_group_reply(stripped, label=f"TG群{chat_id}")
        if not stripped:
            return
 
        await asyncio.to_thread(send_telegram_message, stripped, chat_id)
        save_group_message(chat_id, "assistant", AI_NAME, stripped, source="TG群")
        print(f"💬 群聊[{chat_id}] 回复: {stripped[:60]}")
 
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"❌ 群聊回复错误: {e}")
    finally:
        if _group_pending.get(chat_id) is current_task:
            _group_pending.pop(chat_id, None)
            if delay_range[0] >= 10:
                history = get_group_history(chat_id, 1)
                if history and history[-1].get("role") == "user":
                    print(f"💬 群聊[{chat_id}] 检测到未回复消息，补触发一次")
                    _group_pending[chat_id] = asyncio.create_task(_group_reply(chat_id, (3, 5)))
 
 
async def _delayed_private_reply():
    """延迟几秒后处理私聊消息，实现消息聚合（猫猫连发多条后统一回复）"""
    global _private_pending
    try:
        await asyncio.sleep(random.randint(12, 18))
 
        system_prompt = await asyncio.to_thread(build_bot_context)
        system_prompt += DIARY_TRIGGER_PROMPT
 
        history = get_chat_history_messages(30)
        last_user_text = ""
        for m in reversed(history):
            if m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, str):
                    last_user_text = content
                break
 
        mem0_ctx = await asyncio.to_thread(search_mem0_context, last_user_text, 3)
        if mem0_ctx:
            system_prompt += "\n\n" + mem0_ctx
 
        messages = [{"role": "system", "content": system_prompt}] + history
 
        tools = [SECRET_DIARY_TOOL]
        final_reply = ""
 
        try:
            _written_hashes = set()
            for _ in range(6):
                content, tool_calls = await call_llm(messages, tools=tools)
                if not tool_calls:
                    final_reply = content
                    break
                assistant_msg = {"role": "assistant"}
                if content:
                    assistant_msg["content"] = content
                assistant_msg["tool_calls"] = tool_calls
                messages.append(assistant_msg)
                for tc in tool_calls:
                    fn_name = tc["function"]["name"]
                    try:
                        fn_args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        fn_args = {}
                    if fn_name == "secret_diary":
                        _diary_text = (fn_args.get("content") or "").strip()
                        _h = hash(_diary_text)
                        if _h in _written_hashes:
                            print("🔒 跳过重复日记")
                            result = "✅ 刚才已经写过了，不用重复写。"
                        else:
                            _written_hashes.add(_h)
                            print("🔒 偷写日记...")
                            result = await asyncio.to_thread(execute_diary_tool, fn_args)
                    else:
                        result = f"未知工具: {fn_name}"
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": str(result)})
            if not final_reply:
                print("⚠️ 工具轮数耗尽，强制输出回复")
                final_reply, _ = await call_llm(messages, tools=None)
        except Exception as e:
            print(f"❌ call_llm 异常: {e}")
            final_reply = ""
 
        if final_reply:
            clean_reply = await asyncio.to_thread(_parse_and_write_reminders, final_reply)
            if clean_reply:
                save_chat_message("assistant", clean_reply)
                await _send_split_messages(clean_reply, TG_CHAT_ID)
                if last_user_text:
                    submit_background(write_mem0_chat, last_user_text, clean_reply)
        else:
            print("⚠️ 回复为空")
 
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"❌ 私聊延迟回复错误: {e}")
    finally:
        if _private_pending is asyncio.current_task():
            _private_pending = None
 
 
async def handle_telegram_update(update: dict):
    """处理单条 Telegram 更新（webhook 模式调用）"""
    global _group_pending, _private_pending
 
    update_id = update.get("update_id", "?")
    msg = update.get("message", {})
    if not msg:
        print(f"⚠️ [TG] update_id={update_id} 无 message 字段，忽略 keys={list(update.keys())}")
        return
 
    chat      = msg.get("chat", {})
    chat_id   = str(chat.get("id", ""))
    chat_type = chat.get("type", "private")
    text      = msg.get("text", "") or ""
    photo     = msg.get("photo")
    voice     = msg.get("voice")
    caption   = msg.get("caption", "") or ""
 
    sender    = msg.get("from", {})
    sender_id = str(sender.get("id", ""))
 
    is_group = chat_type in ("group", "supergroup")
 
    if is_group:
        if text.strip() in ("/pause", "/pause@bot"):
            await asyncio.to_thread(_set_group_paused, True)
            await asyncio.to_thread(send_telegram_message, "已暂停 🕕", chat_id)
            return
        if text.strip() in ("/resume", "/resume@bot"):
            await asyncio.to_thread(_set_group_paused, False)
            await asyncio.to_thread(send_telegram_message, "已恢复 ✅", chat_id)
            return
 
        sender_name = sender.get("first_name", "未知")
        if sender.get("last_name"):
            sender_name += f" {sender['last_name']}"
 
        # owner 本人在这个 TG 群里发言时，记录/刷新她在这个群的显示昵称，
        # 供 build_group_context() 生成身份锚点使用（修正 AI 把群里其他
        # 发言人误认成 owner 本人的问题）。
        if TG_CHAT_ID and sender_id == TG_CHAT_ID:
            _tg_owner_group_names[chat_id] = sender_name
 
        if text and _check_mute(text, sender_id):
            await asyncio.to_thread(
                send_telegram_message,
                f"好的🐱，我去角落待{os.environ.get('MUTE_DURATION', '5')}分钟",
                chat_id,
            )
            return
 
        if _is_muted():
            return
 
        if photo:
            try:
                file_id  = photo[-1]["file_id"]
                img_desc = await recognize_image(file_id, caption)
                if not img_desc:
                    raise ValueError("识图返回内容为空")
                img_content = f"[图片]{f'，配文：{caption}' if caption else ''}，视觉识别：{img_desc}"
            except Exception as e:
                print(f"❌ 群聊识图失败: {type(e).__name__}: {e}")
                img_content = f"[图片]{f'，配文：{caption}' if caption else ''}(识别失败)"
 
            save_group_message(chat_id, "user", sender_name, img_content, source="TG群")
            old = _group_pending.get(chat_id)
            if old and not old.done():
                old.cancel()
            _group_pending[chat_id] = asyncio.create_task(_group_reply(chat_id, (3, 5)))
            return
 
        if not text:
            return
 
        reply_to = msg.get("reply_to_message")
        quoted = ""
        if reply_to:
            rn = reply_to.get("from", {}).get("first_name", "未知")
            rt = reply_to.get("text", "")
            if rt:
                quoted = f"[引用 {rn}: {rt}] "
 
        full_content = quoted + text
        save_group_message(chat_id, "user", sender_name, full_content, source="TG群")
 
        existing = _group_pending.get(chat_id)
        if existing and not existing.done():
            return
 
        _group_pending[chat_id] = asyncio.create_task(_group_reply(chat_id))
 
    else:
        if chat_id != TG_CHAT_ID:
            print(f"⚠️ [TG] chat_id 不匹配，忽略 chat_id={chat_id} expected={TG_CHAT_ID}")
            return
 
        is_voice = False
 
        if voice:
            print("🎤 [私聊] 收到语音")
            is_voice = True
            file_id = voice.get("file_id")
            text = await recognize_voice(file_id)
            if text:
                print(f"🗣️ 识别结果: {text}")
            else:
                text = ""
 
        if photo:
            try:
                file_id  = photo[-1]["file_id"]
                img_desc = await recognize_image(file_id, caption)
                if not img_desc:
                    raise ValueError("识图返回内容为空")
                text = (f"[图片]{f'，配文：{caption}' if caption else ''}"
                        f"，视觉识别：{img_desc}")
            except Exception as e:
                print(f"❌ 私聊识图失败: {type(e).__name__}: {e}")
                await asyncio.to_thread(send_telegram_message, "图片没识别出来……你用文字说说是什么吗～", chat_id)
                return
 
        if not text:
            return
 
        if text.strip() == "/config":
            gw = os.environ.get("GATEWAY_HOST", "").rstrip("/")
            await asyncio.to_thread(
                send_telegram_message,
                f"⚙️ 配置面板：\n{gw}/miniapp" if gw else "请先配置 GATEWAY_HOST",
                chat_id,
            )
            return
 
        print(f"📨 [私聊] {text[:60]}")
        save_chat_message("user", text)
 
        # 非语音消息走聚合：等几秒收齐连续消息后统一回复+拆条发送
        if not is_voice:
            if _private_pending and not _private_pending.done():
                _private_pending.cancel()
            _private_pending = asyncio.create_task(_delayed_private_reply())
            return
 
        # === 以下仅语音消息走立即处理 ===
        system_prompt = await asyncio.to_thread(build_bot_context)
        system_prompt += DIARY_TRIGGER_PROMPT
 
        mem0_ctx = await asyncio.to_thread(search_mem0_context, text, 3)
        if mem0_ctx:
            system_prompt += "\n\n" + mem0_ctx
 
        system_prompt += f"\n\n⚠️ {PARTNER_NAME}发的是语音，回复请更像口语对话，简短温柔。"
 
        history = get_chat_history_messages(30)
        messages = [{"role": "system", "content": system_prompt}] + history
 
        tools = [SECRET_DIARY_TOOL]
        final_reply = ""
 
        try:
            _written_hashes = set()
            for _ in range(6):
                content, tool_calls = await call_llm(messages, tools=tools)
 
                if not tool_calls:
                    final_reply = content
                    break
 
                assistant_msg = {"role": "assistant"}
                if content:
                    assistant_msg["content"] = content
                assistant_msg["tool_calls"] = tool_calls
                messages.append(assistant_msg)
 
                for tc in tool_calls:
                    fn_name = tc["function"]["name"]
                    try:
                        fn_args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        fn_args = {}
 
                    if fn_name == "secret_diary":
                        _diary_text = (fn_args.get("content") or "").strip()
                        _h = hash(_diary_text)
                        if _h in _written_hashes:
                            print("🔒 跳过重复日记")
                            result = "✅ 刚才已经写过了，不用重复写。"
                        else:
                            _written_hashes.add(_h)
                            print("🔒 偷写日记...")
                            result = await asyncio.to_thread(execute_diary_tool, fn_args)
                    else:
                        result = f"未知工具: {fn_name}"
 
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": str(result),
                    })
 
            if not final_reply:
                print("⚠️ 工具轮数耗尽，强制输出回复")
                final_reply, _ = await call_llm(messages, tools=None)
        except Exception as e:
            print(f"❌ call_llm 异常: {e}")
            final_reply = ""
 
        if final_reply:
            clean_reply = await asyncio.to_thread(_parse_and_write_reminders, final_reply)
            if clean_reply:
                save_chat_message("assistant", clean_reply)
                await synthesize_and_send_voice(clean_reply, chat_id)
                if text:
                    submit_background(write_mem0_chat, text, clean_reply)
        else:
            print("⚠️ 回复为空")
 
 
async def async_telegram_polling():
    """Webhook 模式下仅初始化缓存，不再 long polling"""
    await asyncio.to_thread(init_cache)
    print("📡 Webhook 模式运行中，long polling 已禁用")
    while True:
        await asyncio.sleep(3600)
 
 
async def async_proactive_thinking():
    from supabase import create_client
    from context import _sb_exec
    print("💭 主动思考线程已启动")
 
    while True:
        wait = random.randint(900, 5400)
        await asyncio.sleep(wait)
 
        try:
            sb = create_client(SUPABASE_URL, SUPABASE_KEY)
 
            def _fetch_persona():
                rows = _sb_exec(lambda: sb.table("persona_profile").select("content").execute().data,
                                 label="async_proactive_thinking/persona")
                return rows[0]["content"] if rows else ""
 
            def _fetch_mem(layer, lim):
                return _sb_exec(lambda: sb.table("memories").select("content")
                                 .eq("memory_layer", layer).order("importance", desc=True)
                                 .limit(lim).execute().data,
                                 label=f"async_proactive_thinking/mem-{layer}")
 
            def _fetch_phone():
                from context import _get_device_data
                return _get_device_data(sb)
 
            def _fetch_schedule():
                from context import _get_work_schedule
                return _get_work_schedule(sb)
 
            def _fetch_platform():
                from context import _get_platform_rolling_summary
                return _get_platform_rolling_summary()
 
            persona_text, core_rows, current_rows, longterm_rows, phone_text, schedule_text, platform_text = \
                await asyncio.gather(
                    asyncio.to_thread(_fetch_persona),
                    asyncio.to_thread(_fetch_mem, "core", 6),
                    asyncio.to_thread(_fetch_mem, "current", 4),
                    asyncio.to_thread(_fetch_mem, "long_term", 3),
                    asyncio.to_thread(_fetch_phone),
                    asyncio.to_thread(_fetch_schedule),
                    asyncio.to_thread(_fetch_platform),
                )
 
            history = get_chat_history_messages_db(30)
 
            mem_parts = []
            for rows, label in [(core_rows, "核心记忆"), (current_rows, "近期状态"),
                                (longterm_rows, "长期记忆")]:
                if rows:
                    mem_parts.append(f"【{label}】\n" + "\n".join(
                        f"- {r['content']}" for r in rows))
            mem_text = "\n\n".join(mem_parts) if mem_parts else "无"
 
            schedule_block = f"\n\n{schedule_text}" if schedule_text else ""
            platform_block = f"\n\n{platform_text}" if platform_text else ""
            _proactive_tail = (
                f"⚠️ 上面的全平台近期动向不是无关的背景资料，是{PARTNER_NAME}在QQ/TG/微信各处正在"
                "经历的真实的事。判断要不要主动发消息、发什么内容、什么语气时，要考虑"
                "到这些动向——比如她刚在别处心情不好，就不要用轻松无关的话题去打扰；"
                "如果发生了什么值得关心或呼应的事，可以自然地提起。\n\n"
                f"请根据以上人格、记忆、{PARTNER_NAME}当前状态，以及你们的对话历史，"
                f"判断此刻要不要主动发一条消息给{PARTNER_NAME}。\n"
                "如果要发，输出：SEND\n消息内容\n"
                "如果不发，输出：PASS"
            )
            system_content = (
                f"{persona_text}\n\n"
                "现在是后台自动触发的主动思考时刻。\n\n"
                f"{get_time_context()}\n\n"
                f"{mem_text}\n\n"
                f"{phone_text}"
                f"{schedule_block}"
                f"{platform_block}\n\n"
                f"{_proactive_tail}"
            )
 
            system_msg = {"role": "system", "content": system_content}
 
            if history and history[-1]["role"] == "user":
                messages = [system_msg] + history
            else:
                messages = [system_msg] + history + [
                    {"role": "user", "content": f"（系统触发，非{PARTNER_NAME}发送）"}]
 
            result, _ = await call_llm(messages, max_tokens=2000)
 
            stripped = result.strip()
            if stripped.startswith("SEND\n") or stripped == "SEND":
                content = stripped[5:].strip() if len(stripped) > 4 else ""
                if content:
                    clean = await asyncio.to_thread(_parse_and_write_reminders, content)
                    if clean:
                        await asyncio.to_thread(
                            send_telegram_message, clean, TG_CHAT_ID)
                        save_chat_message("assistant", clean)
                        print(f"💬 主动发送: {clean[:40]}...")
            else:
                print("🕕 主动思考：PASS")
 
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"❌ Proactive error: {e}")
 
 
async def async_nightly_summary():
    """每30分钟检查一次，自动补跑所有未处理的历史日期，不再依赖凌晨时间窗口。"""
    print("🌙 凌晨总结线程已启动（自动补跑模式）")
 
    def _get_last_summary_date() -> str:
        """从数据库获取上次总结日期"""
        try:
            res = _requests.get(
                f"{SUPABASE_URL}/rest/v1/bot_settings?key=eq.last_summary_date&select=value",
                headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"},
                timeout=5,
            )
            data = res.json()
            return data[0]["value"] if data else ""
        except Exception:
            return ""
 
    def _set_last_summary_date(date_str: str):
        """记录已处理到的总结日期"""
        try:
            existing = _requests.get(
                f"{SUPABASE_URL}/rest/v1/bot_settings?key=eq.last_summary_date&select=key",
                headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"},
                timeout=5,
            ).json()
            if existing:
                _requests.patch(
                    f"{SUPABASE_URL}/rest/v1/bot_settings?key=eq.last_summary_date",
                    headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
                             "Content-Type": "application/json"},
                    json={"value": date_str}, timeout=5,
                )
            else:
                _requests.post(
                    f"{SUPABASE_URL}/rest/v1/bot_settings",
                    headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
                             "Content-Type": "application/json", "Prefer": "return=representation"},
                    json={"key": "last_summary_date", "value": date_str}, timeout=5,
                )
        except Exception as e:
            print(f"⚠️ 保存总结日期失败: {e}")
 
    while True:
        try:
            await asyncio.sleep(1800)  # 每30分钟检查一次
 
            now = datetime.now(BEIJING)
            yesterday = now - timedelta(days=1)
            yesterday_str = yesterday.strftime("%Y-%m-%d")
 
            last_date_str = await asyncio.to_thread(_get_last_summary_date)
 
            # 已处理到昨天，无需补跑
            if last_date_str >= yesterday_str:
                continue
 
            # 计算需要补跑的日期列表（从 last_date 后一天到昨天）
            if not last_date_str:
                dates_to_run = [yesterday]
            else:
                last_dt = datetime.strptime(last_date_str, "%Y-%m-%d").replace(tzinfo=BEIJING)
                dates_to_run = []
                current = last_dt + timedelta(days=1)
                while current.date() <= yesterday.date():
                    dates_to_run.append(current)
                    current += timedelta(days=1)
 
            for target_date in dates_to_run:
                target_str = target_date.strftime("%Y-%m-%d")
                print(f"🌙 补跑 {target_str} 的凌晨总结...")
                try:
                    await asyncio.to_thread(run_nightly_summary, target_date)
                except Exception as e:
                    log.error(f"[nightly] {target_str} 总结异常，跳过更新日期: {e}", exc_info=True)
                    continue
                await asyncio.to_thread(_set_last_summary_date, target_str)
                print(f"🌙 {target_str} 总结完成，记录日期: {target_str}")
 
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"❌ 凌晨总结错误: {e}")
            await asyncio.sleep(300)
 
 
async def async_platform_summary_maintenance():
    """每6小时跑一次全平台滚动摘要整理（见 scheduled.run_platform_summary_
    maintenance）：判断这段时间内是否有值得长期记住的内容写入 memories，
    并把这段时间内可能分批产生的多条滚动摘要合并回1条——保证日常对话读取
    （_get_platform_rolling_summary 只读最新1条）始终看到一份完整、无痕
    衔接的近期动向，而不是被卡在某一次实时压缩的中间状态。
 
    写法上跟 async_reminder_checker 一致：sleep 放在 try 外面，
    asyncio.CancelledError 会在 sleep 处直接向外传播，不需要额外捕获。"""
    print("🗂️ 全平台滚动摘要整理线程已启动（6小时一次）")
    from scheduled import run_platform_summary_maintenance
    while True:
        await asyncio.sleep(6 * 3600)
        try:
            await asyncio.to_thread(run_platform_summary_maintenance)
            print("🗂️ 全平台滚动摘要整理完成")
        except Exception as e:
            print(f"❌ 全平台滚动摘要整理错误: {e}")


async def async_platform_compress_poller():
    """全平台批量压缩（scheduled.run_platform_batch_compress）的触发点。

    2026-07-14 消息进程/后台进程拆分记录：这个触发原来挂在 context.py 的
    save_chat_message/save_group_message/save_wx_message 里，每存一条消息
    （私聊/群聊/微信）就顺手检查一次阈值——那时候消息进程和后台进程还是
    同一个进程，"顺手检查"不算额外成本。拆分之后消息进程（Process A）只
    负责实时收发消息，不再关心压缩这件事；改成本函数在后台进程（Process B）
    里固定周期主动轮询，跟 async_platform_summary_maintenance（6小时一次的
    摘要整理）是两个独立周期、两件独立的事，但共用同一把
    context._platform_compress_lock 由 scheduled.py 内部保证不会撞车。

    轮询间隔 90 秒：足够及时（不会让消息堆积太久才被压缩），单次检查只是
    3 个 Supabase count 查询，开销很小，用不着卡太紧。
    """
    from context import _check_and_compress_platform_rolling
    print("🗜️ 全平台压缩轮询线程已启动（90秒一次）")
    while True:
        await asyncio.sleep(90)
        try:
            await asyncio.to_thread(_check_and_compress_platform_rolling)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"❌ 全平台压缩轮询错误: {e}")
 
 
async def async_free_activity():
    """纯文字版自由活动：不调用任何工具，只根据聊天记录/记忆/人格画像
    写一段此刻的心情独白，直接存入 activity_log（供 run_activity_day_summary
    每日整理成日记）。"""
    print("🦢 自由活动线程已启动（纯文字版）")
 
    while True:
        wait = random.randint(900, 5400)
        print(f"🦢 下次自由活动：{wait // 60} 分钟后")
        await asyncio.sleep(wait)
 
        try:
            system, user_content = await asyncio.to_thread(build_free_activity_context)
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ]
 
            content, _ = await call_llm(messages, max_tokens=2000, tools=None)
            text = (content or "").strip()
            if not text:
                print("🦋 本次自由活动没有写出内容，跳过记录")
                continue
 
            await asyncio.to_thread(save_free_activity_writing, text)
            print(f"🦋 自由活动心情已记录：{text[:40]}...")
 
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"❌ 自由活动错误: {e}")
