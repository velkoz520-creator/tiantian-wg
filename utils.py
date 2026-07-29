import os
import re
import time
import asyncio
import logging
import threading
import httpx
import requests

TG_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN", "")
TG_CHAT_ID   = os.environ.get("TG_CHAT_ID", "")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

log = logging.getLogger(__name__)

_THINK_TAG_RE = re.compile(r'<think(?:ing)?>.*?</think(?:ing)?>', re.DOTALL)
_THINK_FULLWIDTH_RE = re.compile(
    r'｜ｂｅｇｉｎ＿ｏｆ＿ｔｈｉｎｋｉｎｇ｜.*?｜ｅｎｄ＿ｏｆ＿ｔｈｉｎｋｉｎｇ｜',
    re.DOTALL,
)
_THINK_HALFWIDTH_RE = re.compile(
    r'\|begin_of_thinking\|.*?\|end_of_thinking\|',
    re.DOTALL,
)


def _clean_surrogates(text: str) -> str:
    return text.encode("utf-8", errors="ignore").decode("utf-8")


# ── 群聊回复统一安全过滤 ─────────────────────────────────────
# QQ 群聊（qq_workers.py）和 TG 群聊（workers.py）共用这一份，避免两边
# 各写一份正则、以后改一边忘了改另一边（2026-07-08 排查群聊隐私泄露时
# 发现的真实不一致：TG 群聊有这道关卡，QQ 群聊完全没有）。
_XML_TAG_BLOCK_RE = re.compile(
    r'<(?:function_calls|invoke|antml:|tool_call|parameter)[^>]*>.*?'
    r'</(?:function_calls|invoke|antml:[^>]+|tool_call|parameter)>', re.DOTALL)
_XML_SINGLE_RE = re.compile(
    r'</?(?:function_calls|invoke|antml:[a-z_]+|tool_call|parameter)[^>]*>', re.DOTALL)
_SENSITIVE_LEAK_RE = re.compile(
    r'[0-9]{8,}:[A-Za-z0-9_-]{30,}|sk-[a-zA-Z0-9]{20,}|eyJ[a-zA-Z0-9_-]{20,}'
    r'|supabase\.co|api\.telegram\.org/bot',
    re.IGNORECASE,
)


def sanitize_group_reply(text: str, label: str = "") -> str:
    """群聊回复发出前的最后一道过滤：
    1. 剥掉模型可能误吐出的工具调用 XML 标签残留
    2. 检测到 token/api key/jwt/supabase 域名/telegram bot API 路径这类
       技术凭证特征，直接拦截整条回复（不做局部脱敏，因为没法保证脱干净）
    QQ 群聊、TG 群聊发送前都必须经过这里。
    """
    try:
        text = _XML_TAG_BLOCK_RE.sub('', text)
        text = _XML_SINGLE_RE.sub('', text)
        if _SENSITIVE_LEAK_RE.search(text):
            log.warning("[sanitize_group_reply] 群聊回复疑似泄漏敏感凭证，已拦截 label=%s preview=%r", label, text[:80])
            return ""
        return re.sub(r'\n{3,}', '\n\n', text).strip()
    except Exception as e:
        log.error("[sanitize_group_reply] 过滤异常 label=%s: %s", label, e, exc_info=True)
        return ""


_llm_cfg_cache: tuple[float, dict] | None = None
_llm_cfg_lock = threading.Lock()
_LLM_CFG_TTL = 30


def get_active_llm_config() -> dict:
    """读取当前激活的 LLM 配置，带 30 秒 TTL 缓存。"""
    global _llm_cfg_cache
    now = time.time()
    cached = _llm_cfg_cache
    if cached and now - cached[0] < _LLM_CFG_TTL:
        return cached[1]

    with _llm_cfg_lock:
        cached = _llm_cfg_cache
        if cached and time.time() - cached[0] < _LLM_CFG_TTL:
            return cached[1]

        cfg = None
        try:
            res = requests.get(
                f"{SUPABASE_URL}/rest/v1/llm_config?active=eq.true&limit=1",
                headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"},
                timeout=5,
            )
            data = res.json()
            if data:
                cfg = {
                    "base_url":      data[0]["base_url"].rstrip("/"),
                    "api_key":       data[0]["api_key"],
                    "model":         data[0]["model"],
                    "extra_headers": data[0].get("extra_headers") or {},
                }
        except Exception as e:
            log.warning("[get_active_llm_config] 读取 llm_config 失败，回退环境变量: %s", e, exc_info=True)

        if cfg is None:
            cfg = {
                "base_url":      os.environ.get("CHAT_BASE_URL", "https://relay-cache.sharkielab.com/v1").rstrip("/"),
                "api_key":       os.environ.get("CHAT_API_KEY", ""),
                "model":         os.environ.get("BOT_MODEL", "[特特价次kiro]claude-opus-4-6-thinking"),
                "extra_headers": {},
            }

        _llm_cfg_cache = (time.time(), cfg)
        return cfg


_vision_cfg_cache: tuple[float, dict] | None = None
_vision_cfg_lock = threading.Lock()
_VISION_CFG_TTL = 30


def get_vision_config() -> dict:
    """识图专用配置，带 30 秒 TTL 缓存，逻辑跟 get_active_llm_config 完全一致。

    读取 llm_config 表 vision_active=eq.true 那一行——跟聊天用的 active、
    后台压缩用的 bg_active 是同一张表的三个独立布尔列，互不冲突（各自有
    partial unique index 保证同一时刻只有一行）。没有配置 vision_active
    的行时，回退到 VISION_API_KEY/VISION_BASE_URL/VISION_MODEL_NAME 环境
    变量，跟改造前的行为保持一致，不会因为数据库没配就直接识图失败。
    """
    global _vision_cfg_cache
    now = time.time()
    cached = _vision_cfg_cache
    if cached and now - cached[0] < _VISION_CFG_TTL:
        return cached[1]

    with _vision_cfg_lock:
        cached = _vision_cfg_cache
        if cached and time.time() - cached[0] < _VISION_CFG_TTL:
            return cached[1]

        cfg = None
        try:
            res = requests.get(
                f"{SUPABASE_URL}/rest/v1/llm_config?vision_active=eq.true&limit=1",
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
        except Exception as e:
            log.warning("[get_vision_config] 读取 llm_config(vision_active) 失败，回退环境变量: %s", e)

        if cfg is None:
            cfg = {
                "base_url": os.environ.get("VISION_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
                "api_key": os.environ.get("VISION_API_KEY", ""),
                "model": os.environ.get("VISION_MODEL_NAME", "gpt-4o-mini"),
                "extra_headers": {},
            }

        _vision_cfg_cache = (time.time(), cfg)
        return cfg


def send_telegram_message(text: str, chat_id: str = None):
    chat_id = chat_id or TG_CHAT_ID
    limit = 4000
    chunks = [text[i:i+limit] for i in range(0, len(text), limit)] if text else [""]
    for chunk in chunks:
        last_err: Exception | None = None
        for attempt in range(1, 4):
            try:
                resp = requests.post(
                    f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage",
                    json={"chat_id": chat_id, "text": chunk},
                    timeout=20,
                )
                data = resp.json()
                if data.get("ok"):
                    last_err = None
                    break
                if resp.status_code == 429:
                    retry_after = int(data.get("parameters", {}).get("retry_after", 2 * attempt))
                    last_err = RuntimeError(f"telegram 429 rate limited: {data.get('description')}")
                    log.warning(
                        "[send_telegram_message] 被限流(429) 第%d/3次 chat_id=%s retry_after=%ss",
                        attempt, chat_id, retry_after,
                    )
                    if attempt < 3:
                        time.sleep(min(retry_after, 30))
                    continue
                last_err = RuntimeError(f"telegram ok=false: {data.get('description')}")
                log.warning(
                    "[send_telegram_message] API返回失败 第%d/3次 chat_id=%s chunk_len=%d resp=%s",
                    attempt, chat_id, len(chunk), str(data)[:200],
                )
                if attempt < 3:
                    time.sleep(2 * attempt)
            except Exception as e:
                last_err = e
                log.warning(
                    "[send_telegram_message] 第%d/3次失败 chat_id=%s chunk_len=%d %s: %s",
                    attempt, chat_id, len(chunk), type(e).__name__, e,
                )
                if attempt < 3:
                    time.sleep(2 * attempt)
        if last_err is not None:
            log.error(
                "[send_telegram_message] 3次均失败，放弃 chat_id=%s chunk_len=%d final=%s: %s",
                chat_id, len(chunk), type(last_err).__name__, last_err,
                exc_info=last_err,
            )


def _download_with_retry(url: str, timeout: int, headers: dict | None = None, retries: int = 1) -> bytes:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, timeout=timeout, headers=headers or {})
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            last_exc = e
            if attempt < retries:
                print(f"⚠️ [_download_with_retry] 第{attempt + 1}次下载失败（{type(e).__name__}: {e}），重试中... url={url[:80]}")
                continue
    raise last_exc


async def _call_vision_api(vision_base: str, vision_key: str, payload: dict, retries: int = 1, extra_headers: dict | None = None) -> str:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=90, http2=False) as client:
                resp = await client.post(
                    vision_base + "/chat/completions",
                    headers={"Authorization": f"Bearer {vision_key}", "Content-Type": "application/json", **(extra_headers or {})},
                    json=payload,
                )
                data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last_exc = e
            if attempt < retries:
                print(f"⚠️ [_call_vision_api] 第{attempt + 1}次失败（{type(e).__name__}: {e}），重试中...")
                continue
    raise last_exc


async def recognize_image(file_id: str, caption: str = "") -> str:
    import base64 as _b64

    cfg = await asyncio.to_thread(get_vision_config)
    vision_key   = cfg["api_key"]
    vision_base  = cfg["base_url"]
    vision_model = cfg["model"]
    vision_headers = cfg.get("extra_headers") or {}

    file_info = await asyncio.to_thread(
        lambda: requests.get(
            f"https://api.telegram.org/bot{TG_BOT_TOKEN}/getFile?file_id={file_id}",
            timeout=10,
        ).json()
    )
    file_path = file_info["result"]["file_path"]
    img_url   = f"https://api.telegram.org/file/bot{TG_BOT_TOKEN}/{file_path}"

    img_data = await asyncio.to_thread(
        lambda: _download_with_retry(img_url, timeout=40, retries=1)
    )
    b64_url = "data:image/jpeg;base64," + _b64.b64encode(img_data).decode("utf-8")

    if caption:
        prompt = f"请直接描述这张图片的内容，不需要长篇思考分析过程。如果是梗图或表情包，请说明它的含义。（用户发图时附带的文字：「{caption}」，仅供参考，不要回答这个问题，只描述图片）"
    else:
        prompt = "请直接描述这张图片的内容，不需要长篇思考分析过程。如果是梗图或表情包，请说明它的含义。"

    payload = {
        "model": vision_model,
        "max_tokens": 16000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": b64_url}},
                ],
            }
        ],
    }

    try:
        return await _call_vision_api(vision_base, vision_key, payload, extra_headers=vision_headers)
    except Exception as e:
        print(f"❌ [recognize_image] Vision API 失败: {type(e).__name__}: {e}")
        raise


async def recognize_image_url(url: str, caption: str = "") -> str:
    """直接用图片 URL 识图（供 QQ bot 使用）"""
    import base64 as _b64

    cfg = await asyncio.to_thread(get_vision_config)
    vision_key   = cfg["api_key"]
    vision_base  = cfg["base_url"]
    vision_model = cfg["model"]
    vision_headers = cfg.get("extra_headers") or {}

    img_data = await asyncio.to_thread(
        lambda: _download_with_retry(url, timeout=40, headers={"Referer": "https://qq.com"}, retries=1)
    )
    b64_url = "data:image/jpeg;base64," + _b64.b64encode(img_data).decode("utf-8")

    if caption:
        prompt = f"请直接描述这张图片的内容，不需要长篇思考分析过程。如果是梗图或表情包，请说明它的含义。（用户发图时附带的文字：「{caption}」，仅供参考，不要回答这个问题，只描述图片）"
    else:
        prompt = "请直接描述这张图片的内容，不需要长篇思考分析过程。如果是梗图或表情包，请说明它的含义。"

    payload = {
        "model": vision_model,
        "max_tokens": 16000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": b64_url}},
                ],
            }
        ],
    }

    try:
        return await _call_vision_api(vision_base, vision_key, payload, extra_headers=vision_headers)
    except Exception as e:
        print(f"❌ [recognize_image_url] Vision API 失败: {type(e).__name__}: {e}")
        raise


async def recognize_voice(file_id: str) -> str:
    import time as _time

    sf_key   = os.environ.get("SILICON_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    sf_base  = os.environ.get("SILICON_BASE_URL", "https://api.siliconflow.cn/v1")
    sf_model = os.environ.get("SILICON_STT_MODEL", "FunAudioLLM/SenseVoiceSmall")

    if not sf_key:
        return ""

    def _process():
        from openai import OpenAI

        file_info = requests.get(
            f"https://api.telegram.org/bot{TG_BOT_TOKEN}/getFile?file_id={file_id}",
            timeout=10,
        ).json()
        fp = file_info["result"]["file_path"]
        audio_data = _download_with_retry(
            f"https://api.telegram.org/file/bot{TG_BOT_TOKEN}/{fp}",
            timeout=30, retries=1,
        )

        temp_path = f"/tmp/stt_{int(_time.time())}.ogg"
        with open(temp_path, "wb") as f:
            f.write(audio_data)

        try:
            client = OpenAI(api_key=sf_key, base_url=sf_base)
            with open(temp_path, "rb") as f:
                result = client.audio.transcriptions.create(model=sf_model, file=f)
            text = re.sub(r'[\U00010000-\U0010ffff]', '', result.text).strip()
            return text
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    try:
        return await asyncio.to_thread(_process)
    except Exception as e:
        print(f"❌ 语音识别失败: {e}")
        return ""


async def synthesize_and_send_voice(text: str, chat_id: str = None):
    chat_id = chat_id or TG_CHAT_ID

    def _tts_and_send():
        import subprocess
        import time as _time

        minimax_key = os.environ.get("MINIMAX_API_KEY", "")
        out_mp3 = f"/tmp/tts_{int(_time.time())}.mp3"
        out_ogg = f"/tmp/tts_{int(_time.time())}.ogg"

        try:
            if minimax_key:
                url = "https://api.minimax.chat/v1/t2a_v2"
                headers = {"Authorization": f"Bearer {minimax_key}", "Content-Type": "application/json"}
                payload = {
                    "model": "speech-01-turbo",
                    "text": text[:1200],
                    "stream": False,
                    "voice_setting": {
                        "voice_id": os.environ.get("MINIMAX_VOICE_ID",
                                                    "moss_audio_fd2620f9-bef3-11f0-8647-a697af11f3d9"),
                        "speed": 1.0, "vol": 1.0, "pitch": 0,
                    },
                    "audio_setting": {"sample_rate": 32000, "bitrate": 128000, "format": "mp3"},
                }
                resp = requests.post(url, json=payload, headers=headers, timeout=30)
                if resp.status_code == 404:
                    resp = requests.post(url.replace("minimax.chat", "minimax.io"),
                                         json=payload, headers=headers, timeout=30)
                res_json = resp.json()
                if res_json.get("base_resp", {}).get("status_code") == 0 and "data" in res_json:
                    audio_hex = res_json["data"].get("audio", "")
                    if audio_hex:
                        with open(out_mp3, "wb") as f:
                            f.write(bytes.fromhex(audio_hex))
                    else:
                        print("❌ Minimax 返回音频为空")
                        return
                else:
                    print(f"❌ Minimax 报错: {res_json}")
                    return
            else:
                voice_key  = os.environ.get("VOICE_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
                voice_base = os.environ.get("VOICE_BASE_URL", "https://api.openai.com/v1")
                if not voice_key:
                    print("⚠️ 无 TTS API Key")
                    return
                from openai import OpenAI
                client = OpenAI(api_key=voice_key, base_url=voice_base)
                tts_res = client.audio.speech.create(model="tts-1", voice="echo", input=text[:1200])
                with open(out_mp3, "wb") as f:
                    f.write(tts_res.content)

            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                ffmpeg_exe = "ffmpeg"

            subprocess.run(
                [ffmpeg_exe, "-y", "-i", out_mp3, "-c:a", "libopus", "-b:a", "32k", out_ogg],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

            if os.path.exists(out_ogg):
                with open(out_ogg, "rb") as f:
                    tg_resp = requests.post(
                        f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendVoice",
                        data={"chat_id": chat_id},
                        files={"voice": f},
                        timeout=30,
                    )
                    if tg_resp.json().get("ok"):
                        print("✅ 语音条发送成功")
                    else:
                        print(f"❌ 语音条被拒: {tg_resp.text[:100]}")
            else:
                print("❌ OGG 转换失败")

        finally:
            for p in (out_mp3, out_ogg):
                try:
                    os.remove(p)
                except OSError:
                    pass

    try:
        await asyncio.to_thread(_tts_and_send)
    except Exception as e:
        print(f"❌ TTS 流程报错: {e}")


def _strip_thinking(text: str) -> str:
    text = _THINK_TAG_RE.sub('', text)
    text = _THINK_FULLWIDTH_RE.sub('', text)
    text = _THINK_HALFWIDTH_RE.sub('', text)
    for marker in ('<think>', '<thinking>', '｜ｂｅｇｉｎ＿ｏｆ＿ｔｈｉｎｋｉｎｇ｜', '|begin_of_thinking|'):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return text.strip()


async def call_llm(messages: list, max_tokens: int = 4096, tools: list = None, extra_body: dict = None) -> tuple[str, list]:
    import json as _json

    cfg = await asyncio.to_thread(get_active_llm_config)
    target_url = cfg["base_url"] + "/chat/completions"
    req_headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type":  "application/json",
        **cfg.get("extra_headers", {}),
    }
    payload = {
        "model":      cfg["model"],
        "messages":   messages,
        "max_tokens": max_tokens,
        "stream":     True,
    }
    if tools:
        payload["tools"] = tools
    if extra_body:
        payload.update(extra_body)

    content_buf = []
    tc_map: dict[int, dict] = {}
    _got_done = False

    async with httpx.AsyncClient(timeout=120, http2=False) as client:
        async with client.stream("POST", target_url, headers=req_headers, json=payload) as resp:
            if resp.status_code >= 400:
                body = await resp.aread()
                print(f"❌ LLM error {resp.status_code}: {body.decode()[:200]}")
                return "", []
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    _got_done = True
                    break
                try:
                    chunk = _json.loads(data_str)
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    finish_reason = choices[0].get("finish_reason")
                    if finish_reason:
                        if finish_reason == "length":
                            print(f"⚠️ [call_llm] finish_reason=length，内容被 max_tokens 截断！已收到：「{''.join(content_buf)[:60]}」model={cfg['model']}")
                        else:
                            print(f"✅ [call_llm] finish_reason={finish_reason}")

                    if delta.get("reasoning_content"):
                        continue

                    if delta.get("content"):
                        content_buf.append(delta["content"])

                    for tc_delta in delta.get("tool_calls", []):
                        idx = tc_delta.get("index", 0)
                        if idx not in tc_map:
                            tc_map[idx] = {
                                "id": tc_delta.get("id", ""),
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        tc_entry = tc_map[idx]
                        if tc_delta.get("id"):
                            tc_entry["id"] = tc_delta["id"]
                        fn = tc_delta.get("function", {})
                        if fn.get("name"):
                            tc_entry["function"]["name"] = fn["name"]
                        if fn.get("arguments"):
                            tc_entry["function"]["arguments"] += fn["arguments"]

                except Exception as pe:
                    print(f"⚠️ parse err: {pe}")

    if not _got_done:
        content_preview = "".join(content_buf)[:60].replace("\n", "\\n")
        print(f"⚠️ [call_llm] stream 提前结束，未收到 [DONE]！已收到内容：「{content_preview}」model={cfg['model']}")

    content = _clean_surrogates("".join(content_buf))
    content = _strip_thinking(content)

    tool_calls = [tc_map[i] for i in sorted(tc_map.keys())] if tc_map else []

    if tool_calls:
        print(f"🔧 收到 {len(tool_calls)} 个 tool_call: {[tc['function']['name'] for tc in tool_calls]}")

    return content, tool_calls


async def synthesize_and_send_qq_voice(text: str, target_type: str, target_id: int):
    import base64 as _b64

    def _tts() -> str | None:
        import time as _time

        minimax_key = os.environ.get("MINIMAX_API_KEY", "")
        out_mp3 = f"/tmp/qq_tts_{int(_time.time())}.mp3"

        try:
            if minimax_key:
                url_tts = "https://api.minimax.chat/v1/t2a_v2"
                headers = {"Authorization": f"Bearer {minimax_key}", "Content-Type": "application/json"}
                payload = {
                    "model": "speech-01-turbo",
                    "text": text[:1200],
                    "stream": False,
                    "voice_setting": {
                        "voice_id": os.environ.get(
                            "MINIMAX_VOICE_ID",
                            "moss_audio_fd2620f9-bef3-11f0-8647-a697af11f3d9",
                        ),
                        "speed": 1.0, "vol": 1.0, "pitch": 0,
                    },
                    "audio_setting": {"sample_rate": 32000, "bitrate": 128000, "format": "mp3"},
                }
                resp = requests.post(url_tts, json=payload, headers=headers, timeout=30)
                if resp.status_code == 404:
                    resp = requests.post(
                        url_tts.replace("minimax.chat", "minimax.io"),
                        json=payload, headers=headers, timeout=30,
                    )
                res_json = resp.json()
                if res_json.get("base_resp", {}).get("status_code") == 0 and "data" in res_json:
                    audio_hex = res_json["data"].get("audio", "")
                    if audio_hex:
                        with open(out_mp3, "wb") as f:
                            f.write(bytes.fromhex(audio_hex))
                    else:
                        print("❌ [QQ TTS] Minimax 返回音频为空")
                        return None
                else:
                    print(f"❌ [QQ TTS] Minimax 报错: {res_json}")
                    return None
            else:
                voice_key  = os.environ.get("VOICE_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
                voice_base = os.environ.get("VOICE_BASE_URL", "https://api.openai.com/v1")
                if not voice_key:
                    print("⚠️ [QQ TTS] 无 TTS API Key，跳过")
                    return None
                from openai import OpenAI
                client = OpenAI(api_key=voice_key, base_url=voice_base)
                tts_res = client.audio.speech.create(model="tts-1", voice="echo", input=text[:1200])
                with open(out_mp3, "wb") as f:
                    f.write(tts_res.content)

            with open(out_mp3, "rb") as f:
                return _b64.b64encode(f.read()).decode("utf-8")

        except Exception as e:
            print(f"❌ [QQ TTS] 合成失败: {e}", exc_info=True)
            return None
        finally:
            try:
                os.remove(out_mp3)
            except OSError:
                pass

    try:
        b64_audio = await asyncio.to_thread(_tts)
        if not b64_audio:
            print("❌ [QQ TTS] TTS 结果为空，跳过发送")
            return
        from qq_bot import send_qq_msg_threadsafe
        ok = await asyncio.to_thread(
            send_qq_msg_threadsafe, target_type, target_id,
            f"[CQ:record,file=base64://{b64_audio}]",
        )
        if ok:
            print("✅ [QQ TTS] 语音条发送成功")
        else:
            log.error("[synthesize_and_send_qq_voice] 语音条发送失败 target_type=%s target_id=%s（NapCat 未连接或发送超时）", target_type, target_id)
    except Exception as e:
        log.error("[synthesize_and_send_qq_voice] 发送流程异常 target_type=%s target_id=%s: %s", target_type, target_id, e, exc_info=True)


WX_CDN_BASEURL = os.environ.get("WX_CDN_BASEURL", "https://novac2c.cdn.weixin.qq.com/c2c")


def _wx_parse_aes_key(key_str: str) -> bytes:
    import base64 as _b64
    stripped = key_str.strip()
    if len(stripped) == 32 and all(c in '0123456789abcdefABCDEF' for c in stripped):
        return bytes.fromhex(stripped)
    try:
        raw = _b64.b64decode(stripped)
        if len(raw) == 16:
            return raw
        if len(raw) == 32:
            return bytes.fromhex(raw.decode('ascii'))
    except Exception:
        pass
    raise ValueError(f"[_wx_parse_aes_key] 无法解析 AES key len={len(stripped)}: {stripped[:20]}...")


def _wx_aes_ecb_decrypt(data: bytes, key: bytes) -> bytes:
    from Crypto.Cipher import AES
    from Crypto.Util.Padding import unpad
    cipher = AES.new(key, AES.MODE_ECB)
    return unpad(cipher.decrypt(data), 16)


def _wx_aes_ecb_encrypt(data: bytes, key: bytes) -> bytes:
    from Crypto.Cipher import AES
    from Crypto.Util.Padding import pad
    cipher = AES.new(key, AES.MODE_ECB)
    return cipher.encrypt(pad(data, 16))


async def _wx_cdn_download(encrypt_query_param: str, aes_key_str: str, timeout: int = 30, retries: int = 1) -> bytes:
    cdn_base = WX_CDN_BASEURL.rstrip("/")
    url = f"{cdn_base}/download?encrypted_query_param={encrypt_query_param}"
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout, http2=False) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                encrypted = resp.content
            key = _wx_parse_aes_key(aes_key_str)
            return _wx_aes_ecb_decrypt(encrypted, key)
        except Exception as e:
            last_exc = e
            if attempt < retries:
                print(f"⚠️ [_wx_cdn_download] 第{attempt + 1}次失败（{type(e).__name__}: {e}），重试中... url={url[:80]}")
                continue
    print(f"❌ [_wx_cdn_download] 失败: {type(last_exc).__name__}: {last_exc} url={url[:80]}")
    raise last_exc


async def recognize_wx_image(image_item: dict, caption: str = "") -> str:
    import base64 as _b64
    import traceback as _tb

    cfg = await asyncio.to_thread(get_vision_config)
    vision_key   = cfg["api_key"]
    vision_base  = cfg["base_url"]
    vision_model = cfg["model"]
    vision_headers = cfg.get("extra_headers") or {}

    media = image_item.get("media") or {}
    encrypt_query_param = media.get("encrypt_query_param", "")
    aes_key = media.get("aes_key", "") or image_item.get("aeskey", "")

    if not encrypt_query_param or not aes_key:
        raise ValueError(
            f"[recognize_wx_image] image_item 缺少 CDN 参数: "
            f"encrypt_query_param={bool(encrypt_query_param)} aes_key={bool(aes_key)}"
        )

    try:
        img_data = await _wx_cdn_download(encrypt_query_param, aes_key, timeout=40)
    except Exception as e:
        print(f"❌ [recognize_wx_image] CDN 下载失败: {type(e).__name__}: {e}")
        _tb.print_exc()
        raise

    b64_url = "data:image/jpeg;base64," + _b64.b64encode(img_data).decode()

    if caption:
        prompt = (
            f"请直接描述这张图片的内容，不需要长篇思考分析过程。如果是梗图或表情包，请说明它的含义。"
            f"（用户发图时附带的文字：「{caption}」，仅供参考，不要回答这个问题，只描述图片）"
        )
    else:
        prompt = "请直接描述这张图片的内容，不需要长篇思考分析过程。如果是梗图或表情包，请说明它的含义。"

    payload = {
        "model": vision_model,
        "max_tokens": 16000,
        "messages": [{"role": "user", "content": [
            {"type": "text",      "text": prompt},
            {"type": "image_url", "image_url": {"url": b64_url}},
        ]}],
    }
    try:
        return await _call_vision_api(vision_base, vision_key, payload, extra_headers=vision_headers)
    except Exception as e:
        print(f"❌ [recognize_wx_image] Vision API 失败: {type(e).__name__}: {e}")
        _tb.print_exc()
        raise


async def recognize_wx_voice(voice_item: dict) -> str:
    import traceback as _tb

    stt_text = (voice_item.get("text") or "").strip()
    if stt_text:
        print(f"🎤 [WX语音] 使用微信内置 STT: {stt_text[:40]}")
        return stt_text

    media = voice_item.get("media") or {}
    encrypt_query_param = media.get("encrypt_query_param", "")
    aes_key             = media.get("aes_key", "")
    if not encrypt_query_param or not aes_key:
        print("⚠️ [recognize_wx_voice] voice_item 缺少 CDN 参数，跳过识别")
        return ""

    sf_key   = os.environ.get("SILICON_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    sf_base  = os.environ.get("SILICON_BASE_URL", "https://api.siliconflow.cn/v1")
    sf_model = os.environ.get("SILICON_STT_MODEL", "FunAudioLLM/SenseVoiceSmall")
    if not sf_key:
        print("⚠️ [recognize_wx_voice] 无 STT API Key，跳过")
        return ""

    try:
        audio_data = await _wx_cdn_download(encrypt_query_param, aes_key, timeout=30)
    except Exception as e:
        print(f"❌ [recognize_wx_voice] CDN 下载失败: {type(e).__name__}: {e}")
        _tb.print_exc()
        return ""

    def _stt(data: bytes) -> str:
        import time as _t, subprocess as _sp
        from openai import OpenAI
        ts       = int(_t.time())
        temp_in  = f"/tmp/wx_voice_{ts}.silk"
        temp_out = f"/tmp/wx_voice_{ts}.mp3"
        with open(temp_in, "wb") as f:
            f.write(data)
        submit_path = temp_in
        try:
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                ffmpeg_exe = "ffmpeg"
            r = _sp.run(
                [ffmpeg_exe, "-y", "-i", temp_in, "-ar", "16000", "-ac", "1", temp_out],
                capture_output=True, timeout=30,
            )
            if r.returncode == 0 and os.path.exists(temp_out):
                submit_path = temp_out
            else:
                print(f"⚠️ [recognize_wx_voice] ffmpeg 转码失败 rc={r.returncode}: {r.stderr[:100]}")
        except Exception as fe:
            print(f"⚠️ [recognize_wx_voice] ffmpeg 异常: {type(fe).__name__}: {fe}")
        try:
            client = OpenAI(api_key=sf_key, base_url=sf_base)
            with open(submit_path, "rb") as f:
                resp = client.audio.transcriptions.create(model=sf_model, file=f)
            return re.sub(r'[\U00010000-\U0010ffff]', '', resp.text).strip()
        except Exception as e:
            print(f"❌ [recognize_wx_voice] STT 失败: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            return ""
        finally:
            for p in (temp_in, temp_out):
                try:
                    os.remove(p)
                except OSError:
                    pass

    try:
        return await asyncio.to_thread(_stt, audio_data)
    except Exception as e:
        print(f"❌ [recognize_wx_voice] 识别流程失败: {type(e).__name__}: {e}")
        _tb.print_exc()
        return ""


async def recognize_qq_voice(url: str) -> str:
    import time as _time
    import subprocess

    sf_key   = os.environ.get("SILICON_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    sf_base  = os.environ.get("SILICON_BASE_URL", "https://api.siliconflow.cn/v1")
    sf_model = os.environ.get("SILICON_STT_MODEL", "FunAudioLLM/SenseVoiceSmall")

    if not sf_key:
        return ""

    def _process():
        from openai import OpenAI

        ts       = int(_time.time())
        temp_in  = f"/tmp/qq_voice_{ts}.silk"
        temp_out = f"/tmp/qq_voice_{ts}.mp3"

        try:
            audio_data = _download_with_retry(url, timeout=30, headers={"Referer": "https://qq.com"}, retries=1)
        except Exception as e:
            print(f"❌ [recognize_qq_voice] 下载失败: {e}")
            return ""

        with open(temp_in, "wb") as f:
            f.write(audio_data)

        submit_path = temp_in
        try:
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                ffmpeg_exe = "ffmpeg"
            result = subprocess.run(
                [ffmpeg_exe, "-y", "-i", temp_in, "-ar", "16000", "-ac", "1", temp_out],
                capture_output=True, timeout=30,
            )
            if result.returncode == 0 and os.path.exists(temp_out):
                submit_path = temp_out
        except Exception as e:
            print(f"⚠️ [recognize_qq_voice] ffmpeg 转码失败，用原始文件: {e}")

        try:
            client = OpenAI(api_key=sf_key, base_url=sf_base)
            with open(submit_path, "rb") as f:
                resp = client.audio.transcriptions.create(model=sf_model, file=f)
            return re.sub(r'[\U00010000-\U0010ffff]', '', resp.text).strip()
        except Exception as e:
            print(f"❌ [recognize_qq_voice] STT 失败: {e}", exc_info=True)
            return ""
        finally:
            for p in (temp_in, temp_out):
                try:
                    os.remove(p)
                except OSError:
                    pass

    try:
        return await asyncio.to_thread(_process)
    except Exception as e:
        print(f"❌ [recognize_qq_voice] 识别失败: {e}")
        return ""
