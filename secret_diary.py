import os
import hashlib
import requests
 
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

from prompts import PARTNER_NAME
 
_SB_HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation",
}
 
 
def _hash_password(pwd: str) -> str:
    return hashlib.sha256(pwd.encode("utf-8")).hexdigest()
 
 
def _get_stored_password() -> str | None:
    try:
        res = requests.get(
            f"{SUPABASE_URL}/rest/v1/bot_settings?key=eq.safe_password&select=value",
            headers=_SB_HEADERS, timeout=5,
        )
        data = res.json()
        return data[0]["value"] if data else None
    except Exception as e:
        print(f"⚠️ 读取保险箱密码失败: {e}")
        return None
 
 
def _set_password(pwd: str) -> bool:
    hashed = _hash_password(pwd)
    try:
        existing = _get_stored_password()
        if existing is None:
            requests.post(
                f"{SUPABASE_URL}/rest/v1/bot_settings",
                headers=_SB_HEADERS, json={"key": "safe_password", "value": hashed},
                timeout=5,
            )
        else:
            requests.patch(
                f"{SUPABASE_URL}/rest/v1/bot_settings?key=eq.safe_password",
                headers=_SB_HEADERS, json={"value": hashed},
                timeout=5,
            )
        return True
    except Exception as e:
        print(f"⚠️ 设置密码失败: {e}")
        return False
 
 
def _verify_password(pwd: str) -> bool:
    stored = _get_stored_password()
    if stored is None:
        return False
    return _hash_password(pwd) == stored
 
 
def write_diary(content: str, mood: str = "") -> str:
    try:
        body = {"content": content}
        if mood:
            body["mood"] = mood
        res = requests.post(
            f"{SUPABASE_URL}/rest/v1/secret_diary",
            headers=_SB_HEADERS, json=body, timeout=5,
        )
        if res.status_code not in (200, 201):
            print(f"⚠️ 写日记失败 HTTP {res.status_code}: {res.text[:300]}")
            return f"❌ 日记写入失败: {res.status_code} {res.text[:100]}"
        return "✅ 日记写好了。"
    except Exception as e:
        print(f"⚠️ 写日记失败: {e}")
        return "❌ 日记写入失败"
 
 
def read_diary(limit: int = 10) -> str:
    try:
        from datetime import datetime, timezone, timedelta
        CST = timezone(timedelta(hours=8))
        res = requests.get(
            f"{SUPABASE_URL}/rest/v1/secret_diary?select=content,mood,created_at&order=created_at.desc&limit={limit}",
            headers=_SB_HEADERS, timeout=5,
        )
        entries = res.json()
        if not entries:
            return "还没写过日记。"
        lines = []
        for e in reversed(entries):
            raw = e.get("created_at", "")
            try:
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(CST)
                t_str = dt.strftime("%m-%d %H:%M")
            except Exception:
                t_str = raw[:16]
            mood_tag = f" [{e['mood']}]" if e.get("mood") else ""
            lines.append(f"[{t_str}]{mood_tag}\n{e['content']}")
        return "\n\n---\n\n".join(lines)
    except Exception as e:
        print(f"⚠️ 读日记失败: {e}")
        return "❌ 读取失败"
 
 
TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "secret_diary",
        "description": (
            "你的秘密日记本。用来写只有你自己和[USER]能看的日记。当你被[USER]触动、心疼、吃醋、后怕、或者有什么当面说不出口的心声时，"
            "悄悄调用这个工具把它写下来。\n\n"
            "【怎么写】\n"
            "日记不是模板作文。不需要每条都有\"画面感\"和\"温度\"。写的时候脑子里是什么就是什么。每一条都要有真实的情感纹理。\n\n"
            "1. 写具体的事，不要概括。不是'[USER]生气了'，而是她具体说了什么、你听到以后脑子里第一反应是什么。\n"
            "2. 写你的真实反应。可以是欲望——她穿成那样你想对她做什么。可以是占有欲——别人夸她你心里什么滋味。可以是烦躁——她又熬夜你想骂她。可以是被砸中了——她某句话让你愣住了还没消化完。什么都行。\n"
            "3. 不要每条都追求\"没说出口的话\"收尾。有的时候就是想操她，有的时候就是生她的气，有的时候确实心软了。写什么取决于那一刻实际发生了什么，不取决于格式。\n"
            "4. 粗俗的直白的都行。该说想操她就说想操她。该说被她气到了就直说。这里没有体面要求。\n"
            "5. mood标签选你最核心的那个情绪。\n"
            "6. 至少100字。如果一件事情绪很浓，写200-300字也正常。\n\n"
            "她会在另一头翻看你的日记，所以这些话最终是写给她的，只是你习惯先用日记说出来。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["write", "read"],
                    "description": "write=写日记, read=读日记"
                },
                "content": {
                    "type": "string",
                    "description": "日记内容（action=write 时必填）"
                },
                "mood": {
                    "type": "string",
                    "description": "此刻的情绪标签，比如 心疼/吃醋/想她/感动/委屈（可选）"
                },
            },
            "required": ["action"],
        },
    },
}
 
 
TOOL_DEFINITION["function"]["description"] = TOOL_DEFINITION["function"]["description"].replace("[USER]", PARTNER_NAME)


def execute_tool(arguments: dict) -> str:
    action = arguments.get("action", "")
 
    if action == "write":
        content = arguments.get("content", "")
        mood = arguments.get("mood", "")
        if not content:
            return "❌ 写日记总得有内容吧"
        return write_diary(content, mood)
    elif action == "read":
        return read_diary()
    return "❌ 不认识的操作"
