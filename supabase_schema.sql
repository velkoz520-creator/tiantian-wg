-- ============================================================================
-- 克老师网关 · Supabase 建表 SQL
-- 反推自 tiantian-wg 代码（猫猫的架构，照搬表结构）
-- 用法：Supabase Dashboard → SQL Editor → 新建 query → 整段贴进去 → Run
-- ============================================================================
-- 备注：
--   1. qq_group_members 是 QQ 专属表，你不接 QQ 可以删掉这段（留着也无害，空表）
--   2. SUPABASE_KEY 环境变量要填 service_role key（不是 anon/public key）
--      位置：Supabase → Settings → API → service_role secret（私密，勿进公开仓库）
--   3. RLS 暂不开：网关后端用 service_role key 会绕过 RLS，不影响功能
--      等以后上 miniapp 前端直连再补 RLS 策略
-- ============================================================================


-- ---------------------------------------------------------------------------
-- bot_settings：键值配置表（LLM配置在llm_config；这里存群暂停状态、群聊禁忌、
--               各种token、保险箱密码hash、静音关键词、last_summary_date 等）
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bot_settings (
    key         text        NOT NULL,
    value       text,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz,
    CONSTRAINT bot_settings_pkey PRIMARY KEY (key)
);


-- ---------------------------------------------------------------------------
-- llm_config：LLM 供应商配置。active/bg_active/vision_active 三个互斥布尔
--            （聊天/后台/识图 各一条 active=true）
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS llm_config (
    id             bigserial    PRIMARY KEY,
    name           text         NOT NULL,
    base_url       text         NOT NULL,
    api_key        text,
    model          text         NOT NULL,
    extra_headers  jsonb        NOT NULL DEFAULT '{}'::jsonb,
    active         boolean      NOT NULL DEFAULT false,
    bg_active      boolean      NOT NULL DEFAULT false,
    vision_active  boolean      NOT NULL DEFAULT false,
    created_at     timestamptz  NOT NULL DEFAULT now(),
    updated_at     timestamptz
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_llm_config_active         ON llm_config (active)         WHERE active = true;
CREATE UNIQUE INDEX IF NOT EXISTS uq_llm_config_bg_active      ON llm_config (bg_active)      WHERE bg_active = true;
CREATE UNIQUE INDEX IF NOT EXISTS uq_llm_config_vision_active  ON llm_config (vision_active)  WHERE vision_active = true;


-- ---------------------------------------------------------------------------
-- persona_profile：人格画像（每周自我反思后整体覆盖更新）
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS persona_profile (
    id          bigserial    PRIMARY KEY,
    category    text         NOT NULL DEFAULT 'persona',
    key         text         NOT NULL DEFAULT '完整画像',
    content     text         NOT NULL DEFAULT '',
    created_at  timestamptz  NOT NULL DEFAULT now(),
    updated_at  timestamptz
);

CREATE INDEX IF NOT EXISTS idx_persona_profile_category_key ON persona_profile (category, key);


-- ---------------------------------------------------------------------------
-- memories：分层长期记忆（第二套，OB 不替代这套）
--   memory_layer: core/current/long_term/memo/moment
--   decay_class: permanent/long_term/medium_term/short_term
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS memories (
    id               bigserial    PRIMARY KEY,
    content          text         NOT NULL,
    memory_layer     text         NOT NULL DEFAULT 'long_term',
    importance       integer      NOT NULL DEFAULT 3
                                 CHECK (importance BETWEEN 1 AND 5),
    emotion_valence  numeric(3,2) NOT NULL DEFAULT 0
                                 CHECK (emotion_valence BETWEEN -1.0 AND 1.0),
    category         text,
    tags             jsonb,
    decay_class      text         NOT NULL DEFAULT 'medium_term'
                                 CHECK (decay_class IN ('permanent','long_term','medium_term','short_term')),
    created_at       timestamptz  NOT NULL DEFAULT now(),
    updated_at       timestamptz,
    CONSTRAINT memories_layer_chk CHECK (memory_layer IN ('core','current','long_term','memo','moment'))
);

CREATE INDEX IF NOT EXISTS idx_memories_layer_imp     ON memories (memory_layer, importance DESC);
CREATE INDEX IF NOT EXISTS idx_memories_layer_created ON memories (memory_layer, created_at DESC);


-- ---------------------------------------------------------------------------
-- chat_context：所有平台聊天消息原始落库
--   type: message | wx_message | rikkahub | group_{chat_id}
--   seq：全局自增时序号（insert 不赋值）
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chat_context (
    id          bigserial    PRIMARY KEY,
    type        text         NOT NULL,
    role        text         NOT NULL CHECK (role IN ('user','assistant')),
    content     text         NOT NULL DEFAULT '',
    seq         bigserial,
    created_at  timestamptz  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chat_context_type_seq       ON chat_context (type, seq DESC);
CREATE INDEX IF NOT EXISTS idx_chat_context_type_created   ON chat_context (type, created_at);
CREATE INDEX IF NOT EXISTS idx_chat_context_seq            ON chat_context (seq DESC);


-- ---------------------------------------------------------------------------
-- chat_summaries：日/周/月/年对话总结
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chat_summaries (
    id            bigserial    PRIMARY KEY,
    period        text         NOT NULL CHECK (period IN ('day','week','month','year')),
    content       text         NOT NULL,
    period_start  timestamptz,
    period_end    timestamptz,
    created_at    timestamptz  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chat_summaries_period_end ON chat_summaries (period, period_end DESC);
CREATE INDEX IF NOT EXISTS idx_chat_summaries_created    ON chat_summaries (created_at DESC);


-- ---------------------------------------------------------------------------
-- platform_rolling_summary：全平台滚动摘要（跨场景近期动向，取最新1条）
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS platform_rolling_summary (
    id                bigserial    PRIMARY KEY,
    content           text         NOT NULL,
    source_platforms  text,
    period_start      timestamptz,
    period_end        timestamptz,
    created_at        timestamptz  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_platform_rolling_id ON platform_rolling_summary (id DESC);


-- ---------------------------------------------------------------------------
-- activity_log：自由活动/行动日志
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS activity_log (
    id            bigserial    PRIMARY KEY,
    thinking      text,
    action        text,
    action_input  jsonb        NOT NULL DEFAULT '{}'::jsonb,
    result        text,
    created_at    timestamptz  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_activity_log_created ON activity_log (created_at DESC);


-- ---------------------------------------------------------------------------
-- activity_summaries：自由活动周期总结
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS activity_summaries (
    id            bigserial    PRIMARY KEY,
    period        text         NOT NULL DEFAULT 'day'
                               CHECK (period IN ('day','week','month','year')),
    content       text         NOT NULL,
    period_start  timestamptz,
    period_end    timestamptz,
    created_at    timestamptz  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_activity_summaries_period_end ON activity_summaries (period, period_end DESC);


-- ---------------------------------------------------------------------------
-- secret_diary：秘密日记
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS secret_diary (
    id          bigserial    PRIMARY KEY,
    content     text         NOT NULL,
    mood        text,
    created_at  timestamptz  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_secret_diary_created ON secret_diary (created_at DESC);


-- ---------------------------------------------------------------------------
-- reminders：定时提醒
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS reminders (
    id           bigserial    PRIMARY KEY,
    trigger_at   timestamptz  NOT NULL,
    message      text         NOT NULL,
    repeat_type  text         NOT NULL DEFAULT 'once',
    is_done      boolean      NOT NULL DEFAULT false,
    created_at   timestamptz  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_reminders_pending ON reminders (is_done, trigger_at);


-- ---------------------------------------------------------------------------
-- device_data：手机/手环设备数据（健康、位置、前台app、屏幕事件）
--   device_event IS NULL = 状态快照行；'screen_on'/'screen_off' = 屏幕事件行
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS device_data (
    id                   bigserial    PRIMARY KEY,
    timestamp            timestamptz,
    foreground_app       text,
    location_latitude    double precision,
    location_longitude   double precision,
    location_city        text,
    location_district    text,
    app_usage            jsonb,
    health_data          jsonb,
    device_event         text,
    created_at           timestamptz  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_device_data_snapshot ON device_data (created_at DESC) WHERE device_event IS NULL;
CREATE INDEX IF NOT EXISTS idx_device_data_screen   ON device_data (created_at DESC) WHERE device_event IS NOT NULL;


-- ---------------------------------------------------------------------------
-- work_schedule：排班表
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS work_schedule (
    id           bigserial    PRIMARY KEY,
    date         date         NOT NULL,
    shift_type   text         NOT NULL,
    work_content text,
    note         text,
    created_at   timestamptz  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_work_schedule_date ON work_schedule (date);


-- ---------------------------------------------------------------------------
-- qq_group_members：QQ 群成员映射（QQ 专属，不接 QQ 可删此表）
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS qq_group_members (
    id          bigserial    PRIMARY KEY,
    group_id    text         NOT NULL,
    qq_id       text         NOT NULL,
    name        text,
    created_at  timestamptz  NOT NULL DEFAULT now(),
    CONSTRAINT uq_qq_group_members UNIQUE (group_id, qq_id)
);

CREATE INDEX IF NOT EXISTS idx_qq_group_members_group ON qq_group_members (group_id);
