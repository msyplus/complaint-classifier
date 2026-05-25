"""
客诉智能分类系统 v3.2
Complaint Intelligent Classification & Suggestion System

独立作品 — 多模型 AI 引擎架构
支持: Ollama(本地免费) | DeepSeek V4 | Gemini 2.0 Flash(免费) | Groq(免费)
技术栈: Python + Streamlit + pandas + Plotly + Multi-LLM
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
import os
import json

# ═══════════════════════════════════════════════════════════
# 页面配置
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="客诉智能分类系统 - AI Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════
# 模型定义
# ═══════════════════════════════════════════════════════════

MODELS = {
    "rule-only": {
        "name": "关键词规则引擎",
        "provider": "内置",
        "icon": "🔧",
        "model_id": None,
        "key_required": False,
        "key_name": None,
        "base_url": None,
        "sdk_type": "rule",
        "description": "纯关键词匹配，无需网络/模型/Key，结果立即可见",
        "speed": "⚡ 即时",
        "cost": "免费",
    },
    "ollama-qwen": {
        "name": "Qwen2.5 3B (本地)",
        "provider": "Ollama",
        "icon": "💻",
        "model_id": "qwen2.5:3b",
        "key_required": False,
        "key_name": None,
        "base_url": "http://localhost:11434/v1",
        "sdk_type": "openai",
        "description": "本地运行，完全免费，无需网络",
        "speed": "🚀 快（本地）",
        "cost": "免费",
    },
    "deepseek": {
        "name": "DeepSeek V4",
        "provider": "DeepSeek",
        "icon": "🐋",
        "model_id": "deepseek-chat",
        "key_required": True,
        "key_name": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "sdk_type": "openai",
        "description": "中文能力最强，需注册获取 API Key",
        "speed": "⚡ 较快",
        "cost": "¥1/百万tokens",
    },
    "gemini": {
        "name": "Gemini 2.0 Flash",
        "provider": "Google",
        "icon": "🌐",
        "model_id": "gemini-2.0-flash",
        "key_required": True,
        "key_name": "GEMINI_API_KEY",
        "base_url": None,
        "sdk_type": "gemini",
        "description": "Google 免费额度：1500次/天",
        "speed": "⚡ 较快",
        "cost": "免费（15次/分钟）",
    },
    "groq": {
        "name": "Llama 3.3 70B (Groq)",
        "provider": "Groq",
        "icon": "⚡",
        "model_id": "llama-3.3-70b-versatile",
        "key_required": True,
        "key_name": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "sdk_type": "openai",
        "description": "Groq 免费额度：30次/分钟",
        "speed": "🔥 极快",
        "cost": "免费（30次/分钟）",
    },
}


# ═══════════════════════════════════════════════════════════
# 多模型客户端管理
# ═══════════════════════════════════════════════════════════

def get_client(model_key):
    """根据模型 key 获取对应的 LLM 客户端"""
    config = MODELS[model_key]

    if config["sdk_type"] == "rule":
        return None  # 规则引擎无需客户端

    if config["sdk_type"] == "openai":
        from openai import OpenAI

        if not config["key_required"]:
            if not check_ollama_available() or not check_ollama_model(config["model_id"]):
                return None
            # Ollama 本地模式 — 无需 key
            return OpenAI(
                base_url=config["base_url"],
                api_key="ollama",  # Ollama 不校验 key，但 OpenAI SDK 要求非空
                timeout=20.0,
                max_retries=0,
            )
        else:
            key = os.getenv(config["key_name"], "") or st.session_state.get(f"key_{model_key}", "")
            if not key:
                return None
            return OpenAI(
                base_url=config["base_url"],
                api_key=key,
            )

    elif config["sdk_type"] == "gemini":
        import google.generativeai as genai

        key = os.getenv("GEMINI_API_KEY", "") or st.session_state.get("key_gemini", "")
        if not key:
            return None
        genai.configure(api_key=key)
        return genai.GenerativeModel(config["model_id"])

    return None


@st.cache_data(ttl=30, show_spinner=False)
def check_ollama_available():
    """检测本地 Ollama 是否在运行"""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        result = s.connect_ex(("127.0.0.1", 11434))
        s.close()
        return result == 0
    except Exception:
        return False


@st.cache_data(ttl=30, show_spinner=False)
def check_ollama_model(model_id="qwen2.5:3b"):
    """检查 Ollama 是否已拉取指定模型"""
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            timeout=2.0,
            max_retries=0,
        )
        models = client.models.list()
        available = [m.id for m in models]
        return model_id in available
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════
# LLM 分析引擎（统一接口，多模型分发）
# ═══════════════════════════════════════════════════════════

def llm_classify(text, model_key, client):
    """使用指定模型进行语义分类"""
    if not client or not text or not text.strip():
        return None

    config = MODELS[model_key]
    categories_desc = "\n".join([
        f"- {v['icon']} {k}: {v['description']}" for k, v in CATEGORY_RULES.items()
    ])

    prompt = f"""你是电商平台客诉分析专家。请分析以下客诉，返回严格的 JSON 格式（不要包含 markdown 标记）：

客诉内容："{text}"

分类选项：
{categories_desc}
- 其他：不属于以上分类的客诉

请返回 JSON：
{{
    "category": "分类结果",
    "confidence": 0.85,
    "reasoning": "分类理由（30字内）",
    "sentiment": "愤怒/焦虑/平静",
    "priority": "P0-紧急/P1-重要/P2-普通",
    "is_compound": false,
    "compound_types": [],
    "keywords_extracted": ["关键词"],
    "action_recommendation": "针对性的处理建议和沟通话术（80字内）"
}}

注意：
- 如果客诉同时涉及多个维度，is_compound 设为 true
- priority 判断：涉媒体/法律/举报 → P0；多次催促/长期不处理 → P1；普通 → P2
- 严格按 JSON 格式输出，不要包含 ```json``` 标记"""

    try:
        if config["sdk_type"] == "openai":
            response = client.chat.completions.create(
                model=config["model_id"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=600,
            )
            raw = response.choices[0].message.content.strip()

        elif config["sdk_type"] == "gemini":
            response = client.generate_content(prompt)
            raw = response.text.strip()

        else:
            return None

        # 清理 markdown 标记
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            raw = "\n".join(lines)

        result = json.loads(raw)
        return result

    except json.JSONDecodeError:
        return {
            "category": "其他", "confidence": 0.0,
            "reasoning": f"{config['name']} 响应解析失败",
            "sentiment": "平静", "priority": "P2-普通",
            "is_compound": False, "compound_types": [],
            "keywords_extracted": [], "action_recommendation": "请重试",
        }
    except Exception as e:
        return {
            "category": "其他", "confidence": 0.0,
            "reasoning": f"调用失败: {str(e)[:50]}",
            "sentiment": "平静", "priority": "P2-普通",
            "is_compound": False, "compound_types": [],
            "keywords_extracted": [], "action_recommendation": "请检查模型连接",
        }


def llm_batch_anomaly(df, model_key, client):
    """使用指定模型进行语义级异常聚类"""
    if not client or df.empty:
        return []

    config = MODELS[model_key]
    texts_sample = df["客诉文本"].tolist()[:100]
    texts_block = "\n".join([f"{i+1}. {t[:120]}" for i, t in enumerate(texts_sample)])

    prompt = f"""你是电商客诉监控专家。以下是 {len(texts_sample)} 条客诉。请识别"批量异常"——同一事件短时间内大量出现的情况。

客诉列表：
{texts_block}

返回 JSON 数组（不要 markdown 标记）：
[{{
    "topic": "异常主题",
    "description": "1-2句话描述",
    "affected_count": 10,
    "sample_ids": [1, 3, 5],
    "keywords": ["关键词"],
    "severity": "🔴 红色预警 或 🟠 橙色预警 或 🟡 黄色预警",
    "recommended_action": "建议响应动作"
}}]

要求：
- 只返回 >=3条聚集的异常
- 无异常返回 []
- 严格 JSON 数组格式"""

    try:
        if config["sdk_type"] == "openai":
            response = client.chat.completions.create(
                model=config["model_id"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2, max_tokens=1500,
            )
            raw = response.choices[0].message.content.strip()
        elif config["sdk_type"] == "gemini":
            response = client.generate_content(prompt)
            raw = response.text.strip()
        else:
            return []

        if raw.startswith("```"):
            lines = [l for l in raw.split("\n") if not l.startswith("```")]
            raw = "\n".join(lines)

        anomalies = json.loads(raw)
        return anomalies if isinstance(anomalies, list) else []
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════
# 分类规则引擎（关键词 Baseline — 保留不变）
# ═══════════════════════════════════════════════════════════

CATEGORY_RULES = {
    "退款类": {
        "keywords": ["退款", "退钱", "退费", "退货", "退差价", "赔付", "赔偿", "补偿", "赔", "退", "仅退款", "退一赔三"],
        "icon": "💰",
        "description": "消费者要求退款/退货/赔偿",
    },
    "物流类": {
        "keywords": ["物流", "快递", "发货", "配送", "没收到", "迟迟", "运单", "集运", "包裹", "签收", "中转", "滞留", "延迟", "未收到", "卡在", "积压"],
        "icon": "📦",
        "description": "物流配送相关投诉",
    },
    "商品质量类": {
        "keywords": ["质量", "坏了", "破损", "瑕疵", "假货", "货不对板", "不符", "次品", "有问题的", "烂", "坏的", "变质", "过期", "虚假宣传", "和图片不一样", "材质", "掉色", "氧化", "假冒", "伪劣", "欺诈", "纯度", "含银量"],
        "icon": "⚠️",
        "description": "商品质量/描述不符投诉",
    },
    "服务态度类": {
        "keywords": ["态度", "骂人", "不理", "敷衍", "冷漠", "不处理", "推诿", "踢皮球", "不理人", "回复慢", "不耐烦", "语气", "投诉客服", "挂断"],
        "icon": "😠",
        "description": "客服/商家服务态度投诉",
    },
}

SENTIMENT_RULES = {
    "愤怒": {
        "keywords": ["投诉", "曝光", "315", "12315", "工商", "媒体", "法院", "起诉", "律师", "严重", "太过分", "欺人", "维权", "举报", "举报", "彻查", "不处理"],
        "multiplier": 2.0,
    },
    "焦虑": {
        "keywords": ["着急", "什么时候", "还要等", "能不能", "帮我查", "不放心", "担心", "怎么办", "多次", "再次", "又", "迟迟", "反复"],
        "multiplier": 1.5,
    },
    "平静": {
        "keywords": [],
        "multiplier": 1.0,
    },
}

URGENCY_PATTERNS = {
    "P0-紧急": {
        "keywords": ["曝光", "315", "12315", "工商", "媒体", "法院", "起诉", "微博", "小红书", "抖音", "严重受伤", "死亡", "炸", "集体", "团伙", "举报"],
        "sentiment_required": "愤怒",
        "threshold": 1,
    },
    "P1-重要": {
        "keywords": ["多次", "催促", "升级", "投诉", "再不处理", "几天了", "一周", "半个月", "一个月", "又出了", "反复", "再不"],
        "sentiment_required": None,
        "threshold": 1,
    },
}


# ═══════════════════════════════════════════════════════════
# 关键词引擎（Baseline）
# ═══════════════════════════════════════════════════════════

def keyword_classify(text):
    if not isinstance(text, str) or not text.strip():
        return "其他", 0.0
    text_lower = text.lower()
    scores = {}
    for category, config in CATEGORY_RULES.items():
        score = sum(1 for kw in config["keywords"] if kw in text_lower)
        if score > 0:
            scores[category] = score
    if not scores:
        return "其他", 0.0
    best = max(scores, key=scores.get)
    return best, min(scores[best] / 5, 1.0)


def keyword_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return "平静"
    text_lower = text.lower()
    if sum(1 for kw in SENTIMENT_RULES["愤怒"]["keywords"] if kw in text_lower) >= 1:
        return "愤怒"
    if sum(1 for kw in SENTIMENT_RULES["焦虑"]["keywords"] if kw in text_lower) >= 1:
        return "焦虑"
    return "平静"


def keyword_priority(text, sentiment, amount=None):
    if not isinstance(text, str) or not text.strip():
        return "P2-普通"
    text_lower = text.lower()
    p0_score = sum(1 for kw in URGENCY_PATTERNS["P0-紧急"]["keywords"] if kw in text_lower)
    if p0_score >= 1 and sentiment == "愤怒":
        return "P0-紧急"
    if p0_score >= 2:
        return "P0-紧急"
    if sum(1 for kw in URGENCY_PATTERNS["P1-重要"]["keywords"] if kw in text_lower) >= 1:
        return "P1-重要"
    if amount is not None and isinstance(amount, (int, float)) and amount > 5000:
        return "P1-重要"
    return "P2-普通"


def keyword_suggestion(category, sentiment, priority):
    suggestions = []
    if priority == "P0-紧急":
        suggestions.append("🚨 P0紧急：建议30分钟内响应并升级至值班主管")
    elif priority == "P1-重要":
        suggestions.append("⚠️ P1重要：建议2小时内响应")
    else:
        suggestions.append("✅ P2普通：按标准SOP处理")
    suggestion_map = {
        "退款类": "💰 核对订单与退款政策，符合规则优先走快速退款通道",
        "物流类": "📦 核实物流状态，联系物流商确认，同步告知预计等待时间",
        "商品质量类": "⚠️ 请消费者提供凭证，核实后提供退换货或补偿方案",
        "服务态度类": "🎧 致歉并正面回应消费者情绪，承诺内部核查服务质量",
    }
    if category in suggestion_map:
        suggestions.append(suggestion_map[category])
    if sentiment == "愤怒":
        suggestions.append("💬 话术：先致歉共情，再说明解决方案")
    elif sentiment == "焦虑":
        suggestions.append("💬 话术：先确认进度，给出明确时间节点")
    return "；".join(suggestions)


def keyword_anomaly_detection(df, min_count=3):
    if df.empty or "分类结果" not in df.columns:
        return []
    anomalies = []
    for category, group in df.groupby("分类结果"):
        if len(group) < min_count:
            continue
        all_keywords = []
        for text in group["客诉文本"]:
            if isinstance(text, str):
                for cat_config in CATEGORY_RULES.values():
                    for kw in cat_config["keywords"]:
                        if kw in text:
                            all_keywords.append(kw)
        kw_counter = Counter(all_keywords)
        common = [kw for kw, cnt in kw_counter.most_common(10) if cnt >= min_count]
        if common:
            affected = sum(1 for t in group["客诉文本"] if isinstance(t, str) and any(k in t for k in common))
            if affected >= min_count:
                severity = "🔴 红色预警" if affected >= 10 else "🟠 橙色预警" if affected >= 5 else "🟡 黄色预警"
                anomalies.append({
                    "异常主题": f"{category} - {common[0]}",
                    "分类": category,
                    "关联关键词": "、".join(common[:3]),
                    "影响单量": affected,
                    "预警等级": severity,
                })
    return anomalies


# ═══════════════════════════════════════════════════════════
# 模拟数据生成
# ═══════════════════════════════════════════════════════════

def generate_sample_data():
    """生成 80 条模拟客诉，含 3 个批量异常 + 复合客诉"""
    records = []

    refund = [
        ["R001", "买了一件衣服，穿了一次就起球了，我要退款！", 189, "2026-05-01"],
        ["R002", "在你们平台买了个耳机，用了两天就坏了，能退钱吗", 299, "2026-05-01"],
        ["R003", "买的护肤品过敏了，脸都红了，要求退款赔偿", 399, "2026-05-02"],
        ["R004", "手表买回来就不走针，明显是次品，退一赔三", 1299, "2026-05-02"],
        ["R005", "鞋子码数不对，我要退货退款，你们客服一直不处理", 259, "2026-05-03"],
        ["R006", "买的包包五金件掉色，质量太差了，要求退款", 459, "2026-05-04"],
        ["R007", "电饭煲用了不到一个月就坏了，申请退款被拒，我要去12315投诉", 599, "2026-05-04"],
        ["R008", "蓝牙音箱连不上手机，申请退货退款，已经寄回去了", 199, "2026-05-05"],
        ["R009", "买的零食保质期还有一周就过期了，商家没标注，要求退款", 68, "2026-05-06"],
        ["R010", "衣服吊牌剪了但穿不了，商家不给退，这合理吗？", 329, "2026-05-08"],
    ]
    records.extend(refund)

    logistics = [
        ["L001", "商品已发货但物流信息3天没更新了，快递员电话打不通", 88, "2026-05-01"],
        ["L002", "订单一周了还没发货，催了好几次了", 320, "2026-05-03"],
        ["L003", "快递把包裹弄丢了，商家推卸责任让我自己找快递公司", 445, "2026-05-06"],
        ["L004", "买的生鲜食品快递慢了两天到了都臭了", 128, "2026-05-07"],
        ["L005", "海外购的货卡在海关快半个月了", 899, "2026-05-08"],
        ["L006", "同城快递跑了三天，这效率太低了", 56, "2026-05-12"],
        ["L007", "填错地址了快递已经发出去了怎么办", 233, "2026-05-13"],
        ["L008", "快递员未经同意把包裹放快递柜了，取件码也没发", 166, "2026-05-14"],
    ]
    records.extend(logistics)

    quality = [
        ["Q001", "手机壳和图片完全不一样，颜色差很多，虚假宣传", 39, "2026-05-02"],
        ["Q002", "奶粉打开有一股怪味，怀疑是假货不敢给孩子喝", 288, "2026-05-04"],
        ["Q003", "窗帘布料和描述的厚度完全不符，太薄了", 176, "2026-05-06"],
        ["Q004", "运动鞋鞋底开胶了才穿了一周，这质量太差了", 399, "2026-05-08"],
        ["Q005", "收到的衣服有明显色差，面料也和描述不一样", 219, "2026-05-10"],
        ["Q006", "买的充电宝容量严重虚标，标注20000实际不到5000", 149, "2026-05-11"],
        ["Q007", "茶叶包装精美但喝起来有霉味，怀疑是陈茶翻新", 268, "2026-05-13"],
    ]
    records.extend(quality)

    service = [
        ["S001", "客服态度极差，我说了半天她一点都不耐烦直接挂断了", 99, "2026-05-03"],
        ["S002", "联系客服三次了每次都是机器人回复根本没人理我", 456, "2026-05-05"],
        ["S003", "商家推诿责任明明是质量问题非说是我自己弄坏的", 688, "2026-05-07"],
        ["S004", "客服答应给我回电等了两天都没有任何消息", 345, "2026-05-09"],
        ["S005", "投诉客服经理后态度更差了这种服务我要曝光到网上", 799, "2026-05-11"],
        ["S006", "转接了三个人每个人都要重新描述问题，体验极差", 267, "2026-05-14"],
    ]
    records.extend(service)

    other = [
        ["O001", "怎么修改收货地址？我已经下单了", 66, "2026-05-02"],
        ["O002", "优惠券为什么用不了？显示不符合条件", 120, "2026-05-06"],
        ["O003", "怎么联系商家？我想确认一下尺码再发货", 355, "2026-05-10"],
        ["O004", "发票怎么开？我要电子发票", 89, "2026-05-13"],
        ["O005", "满减活动具体规则是什么？页面写得不清楚", 42, "2026-05-15"],
    ]
    records.extend(other)

    # 批量异常 1：银饰品材质不符（12条）
    batch1 = [
        ("B1-01", "买的银手镯说是999纯银，拿回来一测根本不是，含银量最多60%，这算不算欺诈", 459, "2026-05-15"),
        ("B1-02", "这个银项链掉色也太严重了吧，戴了两天脖子都绿了，根本不是纯银的", 329, "2026-05-15"),
        ("B1-03", "S925银戒指戴了一周就发黑，我以前买的银饰戴一年都不会这样，肯定是假的", 259, "2026-05-15"),
        ("B1-04", "银耳钉收到就有铜锈味，这是银的吗？我要退货退款", 199, "2026-05-15"),
        ("B1-05", "买的银饰套盒里面好几件都有氧化斑点，商家非说是正常现象，这不是忽悠人吗", 599, "2026-05-16"),
        ("B1-06", "这个银镯子上面明明写的S925但检测出来是铜镀银，假冒伪劣！我要去315举报", 499, "2026-05-16"),
        ("B1-07", "银项链的材质跟详情页完全不符，证书也是假的，太坑人了", 389, "2026-05-16"),
        ("B1-08", "买了两对银耳环都掉色，而且掉色之后里面露出来的是红色的，这根本就是铜的", 289, "2026-05-16"),
        ("B1-09", "那个银饰商家太黑了，多个买家都反映有材质不符的问题，平台到底管不管", 359, "2026-05-17"),
        ("B1-10", "又是银饰又是材质不符，最近看到第N个了，平台能不能管管这类商家", 429, "2026-05-17"),
        ("B1-11", "S925银饰套链，收到货根本不是银的，戴了一次就过敏起疹子，要求退款加赔偿", 699, "2026-05-17"),
        ("B1-12", "银饰手镯所谓的纯银承诺完全是虚假宣传，材质检测不过关，建议彻查该类商家", 559, "2026-05-17"),
    ]
    records.extend(batch1)

    # 批量异常 2：台湾集运物流积压（8条）
    batch2 = [
        ("B2-01", "台湾集运的包裹已经等了一个月了还没到，物流显示一直在中转", 320, "2026-05-16"),
        ("B2-02", "集运包裹卡在中转站不动了，客服也联系不上，我的货到底在哪", 450, "2026-05-16"),
        ("B2-03", "台湾流向的物流是不是出了什么问题，集运订单20多天没更新物流了", 380, "2026-05-17"),
        ("B2-04", "三个集运包裹全部积压，问了物流公司说是运力不够，你们有解决方案吗", 890, "2026-05-17"),
        ("B2-05", "集运台湾的订单物流已经超过30天，打了好多次电话都说在处理", 550, "2026-05-17"),
        ("B2-06", "我的集运包裹显示异常，问客服说在协调但等了一周没任何进展", 420, "2026-05-18"),
        ("B2-07", "台湾集运商到底什么时候能恢复，我的订单各种节日礼物等着用呢", 650, "2026-05-18"),
        ("B2-08", "物流积压这么严重，平台至少应该主动通知消费者，而不是让我们自己发现", 310, "2026-05-18"),
    ]
    records.extend(batch2)

    # 批量异常 3：婴幼儿纸尿裤质量问题（6条）
    batch3 = [
        ("B3-01", "买的纸尿裤宝宝用了红屁股，同一批次的几个妈妈都反映有这个问题", 156, "2026-05-18"),
        ("B3-02", "某品牌纸尿裤这次的质量明显有问题，吸水性比以前差太多了，尿了两三次就侧漏", 128, "2026-05-18"),
        ("B3-03", "婴幼儿纸尿裤打开有一股刺鼻的塑料味，不敢给孩子用了，怀疑使用了问题原材料", 199, "2026-05-18"),
        ("B3-04", "纸尿裤粘连处容易断开，宝宝一翻就散了，之前买的同品牌完全没这个问题", 168, "2026-05-19"),
        ("B3-05", "我家娃用了这个纸尿裤也红屁股了，而且表层还有颗粒物，摸起来很粗糙", 138, "2026-05-19"),
        ("B3-06", "这一批次的纸尿裤明显偷工减料变薄了，价格还没变，这不是割韭菜吗", 145, "2026-05-19"),
    ]
    records.extend(batch3)

    # 复合客诉
    compound = [
        ["M001", "买的银手镯说是纯银结果是假货，我要退款！客服还一直不理人，说什么在核实，都核实三天了", 599, "2026-05-18"],
        ["M002", "台湾集运包裹丢件了，申请理赔客服推三阻四，物流慢也就算了服务还这么差", 780, "2026-05-19"],
        ["M003", "买的纸尿裤质量有问题宝宝红屁股了，想退货退款客服态度极其恶劣说是我自己护理不当", 199, "2026-05-19"],
        ["M004", "快递把电子产品包裹摔坏了，里面的平板屏幕碎了，申请赔偿商家和物流互相推卸责任，我要去小红书曝光", 2499, "2026-05-20"],
    ]
    records.extend(compound)

    return pd.DataFrame(records, columns=["complaint_id", "complaint_text", "order_amount", "create_time"])


# ═══════════════════════════════════════════════════════════
# UI 组件
# ═══════════════════════════════════════════════════════════

def init_session():
    defaults = {
        "working_df": None,
        "selected_model": "rule-only",
        "key_deepseek": "",
        "key_gemini": "",
        "key_groq": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ═══════════════════════════════════════════════════════════
# 版本历史
# ═══════════════════════════════════════════════════════════

VERSION_HISTORY = [
    {
        "version": "v3.2",
        "date": "2026-05-20",
        "title": "分析深度增强 — 双引擎对比报告 + 分歧案例 + 低置信告警",
        "changes": [
            "新增双引擎对比报告Tab：一致率统计、分类交叉矩阵、置信度分布直方图",
            "新增分歧案例专题：自动筛选双引擎分类不一致的case并分析原因",
            "新增低置信度告警：AI置信度<0.6的客诉高亮标记，提示人工审核",
        ],
        "advantage": "不仅做分类，更深入评估分类质量，展示模型评估和Badcase归因能力",
        "icon": "📊",
    },
    {
        "version": "v3.1",
        "date": "2026-05-20",
        "title": "多模型 AI 引擎架构",
        "changes": [
            "新增 Ollama 本地模型支持（Qwen2.5 3B），完全免费无需 API Key",
            "新增 Gemini 2.0 Flash 模型接入（Google 免费额度）",
            "新增 Groq Llama 3.3 模型接入（免费 30次/分钟）",
            "保留 DeepSeek V4 模型选项",
            "侧边栏模型选择器，一键切换四种 AI 引擎",
            "自动检测本地 Ollama 运行状态和模型可用性",
        ],
        "advantage": "多模型灵活切换，本地+云端全覆盖，可按需选择最合适的引擎",
        "icon": "🔄",
    },
    {
        "version": "v3.0",
        "date": "2026-05-20",
        "title": "双引擎对比架构",
        "changes": [
            "接入 DeepSeek V4 实现真正的 LLM 语义分类（替代占位符开关）",
            "新增双引擎并排对比模式：关键词规则 vs AI 语义",
            "LLM 生成个性化处理建议（替代固定模板话术）",
            "新增 AI 语义聚类批量异常检测",
            "增强 Demo 数据至 80 条，新增纸尿裤质量风波异常事件",
            "新增复合型客诉识别（一单多维度）",
        ],
        "advantage": "并排对比直观呈现 AI 语义理解 vs 关键词规则的差异",
        "icon": "🤖",
    },
    {
        "version": "v2.0",
        "date": "2026-05-17",
        "title": "关键词规则引擎版",
        "changes": [
            "基于服务运营经验沉淀的 5 类关键词分类规则",
            "三级优先级评估体系（P0紧急/P1重要/P2普通）",
            "三种情绪识别（愤怒/焦虑/平静）",
            "基于关键词频率的批量异常检测",
            "Plotly 可视化看板（饼图、柱状图、热力图）",
            "CSV 文件上传/下载功能",
        ],
        "advantage": "规则透明可解释，运行速度快（毫秒级），无需任何外部依赖即可完成全部分析",
        "icon": "📊",
    },
    {
        "version": "v1.0",
        "date": "2026-05-17",
        "title": "快速原型版",
        "changes": [
            "Streamlit 基础框架搭建",
            "简单关键词匹配分类逻辑",
            "模拟数据生成器",
            "基础 UI 布局",
        ],
        "advantage": "从 0 到 1 验证了客诉分类的产品方向，确定了功能边界和技术选型",
        "icon": "🚀",
    },
]


def show_version_history():
    """渲染版本历史时间线"""
    for entry in VERSION_HISTORY:
        is_current = entry == VERSION_HISTORY[0]
        border_style = "2px solid #4CAF50" if is_current else "1px solid #444"
        with st.container(border=True):
            cols = st.columns([0.05, 0.95])
            with cols[0]:
                st.markdown(f"### {entry['icon']}")
            with cols[1]:
                st.markdown(
                    f"**{entry['version']}** — {entry['title']} "
                    f"{'`当前版本`' if is_current else ''}"
                )
                st.caption(f"📅 {entry['date']}")
            for change in entry["changes"]:
                st.markdown(f"  • {change}")
            with st.expander("💡 核心优势"):
                st.info(entry["advantage"])


def show_rules_config():
    """侧边栏规则配置面板"""
    with st.expander("📋 规则配置", expanded=False):
        st.caption("调整关键词规则引擎的匹配词库，修改后即时生效")

        # 分类规则
        st.markdown("**分类关键词**")
        for cat, cfg in CATEGORY_RULES.items():
            key = f"rule_cat_{cat}"
            if key not in st.session_state:
                st.session_state[key] = "、".join(cfg["keywords"])
            with st.expander(f"{cfg['icon']} {cat}（{cfg['description']}）", expanded=False):
                new_val = st.text_area(
                    f"关键词（用顿号分隔）",
                    value=st.session_state[key],
                    key=f"input_{key}",
                    height=68,
                    label_visibility="collapsed",
                )
                st.session_state[key] = new_val
                kw_list = [kw.strip() for kw in new_val.replace(",", "、").split("、") if kw.strip()]
                CATEGORY_RULES[cat]["keywords"] = kw_list
                st.caption(f"共 {len(kw_list)} 个关键词")

        # 情绪规则
        st.divider()
        st.markdown("**情绪识别关键词**")
        for sent, cfg in SENTIMENT_RULES.items():
            if sent == "平静" and not cfg["keywords"]:
                continue
            key = f"rule_sent_{sent}"
            if key not in st.session_state:
                st.session_state[key] = "、".join(cfg["keywords"])
            with st.expander(f"{'🔴' if sent == '愤怒' else '🟠'} {sent}（权重 {cfg['multiplier']}x）", expanded=False):
                new_val = st.text_area(
                    f"关键词_{sent}",
                    value=st.session_state[key],
                    key=f"input_{key}",
                    height=50,
                    label_visibility="collapsed",
                )
                st.session_state[key] = new_val
                kw_list = [kw.strip() for kw in new_val.replace(",", "、").split("、") if kw.strip()]
                SENTIMENT_RULES[sent]["keywords"] = kw_list

        # 紧急关键词
        st.divider()
        st.markdown("**优先级判定关键词**")
        for pri, cfg in URGENCY_PATTERNS.items():
            key = f"rule_pri_{pri}"
            if key not in st.session_state:
                st.session_state[key] = "、".join(cfg["keywords"])
            req = f"（需同时满足: {cfg['sentiment_required']}）" if cfg["sentiment_required"] else ""
            with st.expander(f"{'🔴' if 'P0' in pri else '🟠'} {pri}{req}", expanded=False):
                new_val = st.text_area(
                    f"关键词_{pri}",
                    value=st.session_state[key],
                    key=f"input_{key}",
                    height=50,
                    label_visibility="collapsed",
                )
                st.session_state[key] = new_val
                kw_list = [kw.strip() for kw in new_val.replace(",", "、").split("、") if kw.strip()]
                URGENCY_PATTERNS[pri]["keywords"] = kw_list

        # 重置
        st.divider()
        if st.button("🔄 恢复默认规则", width='stretch'):
            for cat, cfg in CATEGORY_RULES.items():
                st.session_state.pop(f"rule_cat_{cat}", None)
            for sent, cfg in SENTIMENT_RULES.items():
                st.session_state.pop(f"rule_sent_{sent}", None)
            for pri, cfg in URGENCY_PATTERNS.items():
                st.session_state.pop(f"rule_pri_{pri}", None)
            st.rerun()


# ═══════════════════════════════════════════════════════════
# 问题反馈入口
# ═══════════════════════════════════════════════════════════

ISSUE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "issue_reports.jsonl")


def save_issue(issue_data):
    """保存问题报告到本地 JSONL"""
    issue_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ISSUE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(issue_data, ensure_ascii=False) + "\n")


def show_issue_report():
    """侧边栏问题反馈入口"""
    with st.expander("🐛 问题反馈", expanded=False):
        st.caption("发现 bug、规则不准、体验问题？请在这里提交")

        issue_title = st.text_input("问题标题", key="issue_title", placeholder="简要用一句话描述问题")
        issue_desc = st.text_area(
            "详细描述",
            key="issue_desc",
            height=80,
            placeholder="描述：操作了什么、预期什么结果、实际发生了什么...",
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            issue_type = st.selectbox(
                "问题类型",
                ["分类不准确", "系统Bug", "功能建议", "数据问题", "界面体验", "其他"],
                key="issue_type",
            )
        with c2:
            issue_urgency = st.selectbox(
                "紧急程度",
                ["一般", "重要", "紧急"],
                key="issue_urgency",
            )

        if st.button("提交问题", key="issue_submit", type="primary", width='stretch'):
            if not issue_title.strip():
                st.error("请填写问题标题")
            else:
                issue = {
                    "title": issue_title,
                    "description": issue_desc,
                    "type": issue_type,
                    "urgency": issue_urgency,
                }
                save_issue(issue)
                st.success("问题已提交，感谢反馈！")
                # 清空
                st.session_state["issue_title"] = ""
                st.session_state["issue_desc"] = ""
                st.rerun()


def show_welcome():
    st.markdown("""
    ### 👋 VOC 智能分类与优先级评估 v3.2

    这个 Demo 模拟服务 AI 工作流的第一步：把一条客诉/VOC 文本转化为**类别、情绪、优先级、处理建议和批量异常信号**。

    **默认无需 API Key**：关键词规则引擎可直接演示；接入本地 Ollama、DeepSeek、Gemini 或 Groq 后可切换为 AI 增强分析。

    ---

    #### 🎯 双引擎架构

    | 模块 | 关键词引擎（Rule-based） | AI 引擎（多模型可选） |
    |------|------------------------|----------------------|
    | 分类方式 | 关键词硬匹配，5个预设类别 | 语义理解，支持复合分类 |
    | 情绪识别 | 关键词触发 | 上下文语义判断 |
    | 优先级 | 关键词+金额规则 | 综合语义+上下文评估 |
    | 处理建议 | 模板化固定话术 | 针对原文本的个性化建议 |
    | 异常检测 | 关键词频率聚类 | 语义级事件聚类 |

    #### 🚀 面试演示路径

    1. 直接点击侧边栏 **加载 80 条模拟客诉数据**。
    2. 先用默认规则引擎展示稳定结果。
    3. 如有 API Key 或本地 Ollama，再切换 AI 引擎对比语义分析效果。
    """)
    st.info("👈 在侧边栏选择分析引擎 → 加载数据 → 查看分析结果")


def show_model_selector():
    """模型选择器 UI"""
    st.subheader("🤖 分析引擎选择")

    # 检测 Ollama
    ollama_running = check_ollama_available()
    ollama_has_model = check_ollama_model() if ollama_running else False

    # 可用的模型选项（按可用性排序）
    model_options = {}
    model_status = {}

    # 规则引擎始终可用
    model_options["rule-only"] = f"🔧 关键词规则引擎（默认·即时可用）"
    model_status["rule-only"] = "✅ 已就绪"

    for key, config in MODELS.items():
        if key == "rule-only":
            continue  # 已在上面处理
        elif key == "ollama-qwen":
            if ollama_running and ollama_has_model:
                model_options[key] = f"{config['icon']} {config['name']}"
                model_status[key] = "✅ 已就绪"
            elif ollama_running:
                model_options[key] = f"{config['icon']} {config['name']} (需拉取模型)"
                model_status[key] = "⚠️ 模型未拉取"
            else:
                model_options[key] = f"{config['icon']} {config['name']} (Ollama 未启动)"
                model_status[key] = "❌ 未启动"
        elif key == "deepseek":
            model_options[key] = f"{config['icon']} {config['name']} (需 API Key)"
            model_status[key] = "🔑 需 Key" if not st.session_state.get("key_deepseek") else "✅ 已配置"
        elif key == "gemini":
            model_options[key] = f"{config['icon']} {config['name']} (免费)"
            model_status[key] = "🔑 需 Key" if not st.session_state.get("key_gemini") else "✅ 已配置"
        elif key == "groq":
            model_options[key] = f"{config['icon']} {config['name']} (免费)"
            model_status[key] = "🔑 需 Key" if not st.session_state.get("key_groq") else "✅ 已配置"

    # 选择模型 — 默认规则引擎
    current = st.session_state.get("selected_model", "rule-only")
    if current not in model_options:
        current = "rule-only"

    selected_label = st.selectbox(
        "选择分析引擎",
        options=list(model_options.keys()),
        format_func=lambda k: model_options[k],
        index=list(model_options.keys()).index(current),
        key="model_selector",
    )
    st.session_state["selected_model"] = selected_label
    config = MODELS[selected_label]

    with st.container(border=True):
        st.caption(f"**{config['icon']} {config['name']}** | {config['provider']} | {config['speed']} | {config['cost']}")
        st.caption(config["description"])
        st.caption(f"状态: {model_status.get(selected_label, '')}")

    # API Key 输入（仅需要的模型）
    if config["key_required"]:
        key_label = f"{config['provider']} API Key"
        key_value = st.text_input(
            key_label,
            type="password",
            value=st.session_state.get(f"key_{selected_label}", ""),
            placeholder=f"输入 {config['provider']} API Key...",
            key=f"api_key_{selected_label}",
        )
        if key_value:
            st.session_state[f"key_{selected_label}"] = key_value

    # Ollama 状态提示
    if selected_label == "ollama-qwen":
        if not ollama_running:
            st.warning("⚠️ Ollama 未运行。请先安装并启动: https://ollama.com/download/windows")
            st.code("ollama pull qwen2.5:3b   # 拉取模型（仅需一次）\nollama serve             # 启动服务", language="bash")
        elif not ollama_has_model:
            st.warning("⚠️ Qwen2.5 模型未拉取。运行: `ollama pull qwen2.5:3b`")

    return selected_label


def show_sidebar():
    """完整侧边栏"""
    with st.sidebar:
        st.header("⚙️ 控制面板")

        # 模型选择
        selected_model = show_model_selector()
        config = MODELS[selected_model]

        st.divider()

        # 数据加载
        st.subheader("📤 数据加载")
        uploaded_file = st.file_uploader("上传客诉工单 CSV", type=["csv"], help="CSV 需包含 complaint_text 列")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                text_col = None
                for col in ["complaint_text", "客诉文本", "投诉内容", "voc_text", "content", "text"]:
                    if col in df.columns:
                        text_col = col
                        break
                if text_col is None:
                    for col in df.columns:
                        if df[col].dtype == "object" and df[col].str.len().mean() > 20:
                            text_col = col
                            break
                if text_col:
                    df["客诉文本"] = df[text_col].astype(str)
                    st.session_state["working_df"] = df
                    st.success(f"已加载 {len(df)} 条数据")
                else:
                    st.error("未找到客诉文本列")
                    st.info("请确认 CSV 至少包含以下任一列名：complaint_text、客诉文本、投诉内容、voc_text、content、text。")
            except Exception as e:
                st.error(f"文件读取失败: {e}")
                st.info("建议先使用右侧「加载 80 条模拟客诉数据」完成演示；上传文件请使用 UTF-8 CSV，并保留文本列。")

        st.divider()

        # 示例数据
        st.subheader("📥 快速体验")
        btn_label = "🎲 加载 80 条模拟客诉数据"
        if st.button(btn_label, type="primary", width='stretch'):
            with st.spinner("生成模拟数据..."):
                st.session_state["working_df"] = generate_sample_data()
            st.rerun()

        if st.session_state.get("working_df") is not None:
            if st.button("🗑️ 清除数据", width='stretch'):
                st.session_state["working_df"] = None
                st.rerun()

        st.divider()

        # 模型配置信息
        st.subheader("📋 当前模型信息")
        with st.container(border=True):
            st.markdown(f"**{config['icon']} {config['name']}**")
            st.caption(f"提供商: {config['provider']}")
            st.caption(f"模型: {config['model_id']}")
            st.caption(f"速度: {config['speed']}")
            st.caption(f"费用: {config['cost']}")

        st.divider()
        st.caption("💡 推荐安装 Ollama 本地运行，完全免费无需 Key")

        # 规则配置
        show_rules_config()
        # 反馈汇总
        show_issue_report()


        # 版本历史（折叠）
        with st.expander("📜 版本演进历史", expanded=False):
            for entry in VERSION_HISTORY:
                st.markdown(
                    f"{entry['icon']} **{entry['version']}** — {entry['title']}"
                    f"{' `当前`' if entry == VERSION_HISTORY[0] else ''}"
                )
                st.caption(f"📅 {entry['date']}  |  {len(entry['changes'])} 项改动")
                st.caption(f"_{entry['advantage'][:50]}..._")
                if entry != VERSION_HISTORY[-1]:
                    st.markdown("│")

        return selected_model


def show_batch_analysis(df, model_key, client):
    """批量分析结果展示"""
    # KPI
    st.subheader("📊 分析概览")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("客诉总数", len(df))
    with c2:
        p0 = len(df[df["优先级"] == "P0-紧急"])
        st.metric("P0 紧急", p0, delta="需立即处理" if p0 > 0 else None)
    with c3:
        p1 = len(df[df["优先级"] == "P1-重要"])
        st.metric("P1 重要", p1)
    with c4:
        anger = len(df[df["情绪"] == "愤怒"])
        st.metric("愤怒情绪", anger)
    with c5:
        n_cat = df["分类结果"].nunique()
        st.metric("涉及分类", n_cat)

    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📋 分类结果明细", "📊 对比报告", "📈 统计分析", "🔍 批量异常检测", "📥 导出"]
    )

    with tab1:
        show_result_table(df)

    with tab2:
        show_comparison_report(df)

    with tab3:
        show_analytics(df)

    with tab4:
        show_anomaly(df, model_key, client)

    with tab5:
        show_export(df)


def show_comparison_report(df):
    """双引擎对比报告：一致率、分歧分析、置信度分布"""
    st.subheader("双引擎对比分析报告")
    has_ai = "AI分类结果" in df.columns
    if not has_ai:
        st.info("切换到 AI 引擎后可查看双引擎对比报告")
        return
    match_mask = df["分类结果"] == df["AI分类结果"]
    match_rate = match_mask.sum() / len(df) * 100
    divergent = df[~match_mask]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("双引擎一致率", f"{match_rate:.1f}%",
                  delta="高一致" if match_rate >= 80 else "需关注" if match_rate >= 60 else "偏差大")
    with c2:
        st.metric("分歧案例", len(divergent), delta=f"{len(divergent)/len(df)*100:.0f}%")
    with c3:
        low_conf = (df["AI置信度"] < 0.6).sum() if "AI置信度" in df.columns else 0
        st.metric("低置信度(<0.6)", low_conf, delta="模糊地带")
    st.subheader("分类交叉矩阵（规则 vs AI）")
    cross = pd.crosstab(df["分类结果"], df["AI分类结果"])
    fig = px.imshow(cross.values, x=cross.columns, y=cross.index,
                    title="规则引擎 vs AI 引擎分类对比", color_continuous_scale="Blues", text_auto=True)
    st.plotly_chart(fig, width='stretch')
    if "AI置信度" in df.columns:
        st.subheader("AI 置信度分布")
        fig_hist = px.histogram(df, x="AI置信度", nbins=10, title="AI 分类置信度分布",
                                color_discrete_sequence=["#2196F3"])
        fig_hist.add_vline(x=0.6, line_dash="dash", line_color="red", annotation_text="低置信阈值")
        st.plotly_chart(fig_hist, width='stretch')
    if len(divergent) > 0:
        st.subheader(f"分歧案例专题（{len(divergent)} 条）")
        for _, row in divergent.head(15).iterrows():
            kw_cat = row["分类结果"]
            ai_cat = row.get("AI分类结果", "?")
            ai_reason = row.get("LLM分类理由", "")
            ai_conf = row.get("AI置信度", 0)
            text = str(row.get("客诉文本", ""))[:100]
            icon = "🤖" if ai_conf >= 0.7 else "❓"
            with st.expander(f"{icon} 规则={kw_cat} -> AI={ai_cat} | 置信度{ai_conf:.0%} | {text[:50]}..."):
                st.markdown(f"**原文**: {text}")
                st.markdown(f"**关键词**: {kw_cat}  |  **AI**: {ai_cat}（{ai_reason}）")
                if ai_conf < 0.6:
                    st.warning("AI 置信度低，建议人工审核")
    if "AI置信度" in df.columns:
        low_conf_df = df[df["AI置信度"] < 0.6]
        if len(low_conf_df) > 0:
            st.divider()
            st.subheader(f"低置信度告警（{len(low_conf_df)} 条）")
            for _, row in low_conf_df.head(10).iterrows():
                st.markdown(f"- [{row.get('AI置信度', 0):.0%}] {row['分类结果']} | {str(row.get('客诉文本', ''))[:80]}...")


def show_result_table(df):
    st.subheader("客诉分类结果明细")
    cf1, cf2, cf3 = st.columns(3)
    with cf1:
        fc = st.multiselect("分类筛选", df["分类结果"].unique().tolist(), key="fc_batch")
    with cf2:
        fp = st.multiselect("优先级筛选", ["P0-紧急", "P1-重要", "P2-普通"], key="fp_batch")
    with cf3:
        fs = st.multiselect("情绪筛选", ["愤怒", "焦虑", "平静"], key="fs_batch")

    ddf = df.copy()
    if fc:
        ddf = ddf[ddf["分类结果"].isin(fc)]
    if fp:
        ddf = ddf[ddf["优先级"].isin(fp)]
    if fs:
        ddf = ddf[ddf["情绪"].isin(fs)]

    st.caption(f"共 {len(ddf)} 条结果")

    for _, row in ddf.iterrows():
        pc = {"P0-紧急": "red", "P1-重要": "orange", "P2-普通": "green"}
        icon = CATEGORY_RULES.get(row["分类结果"], {}).get("icon", "📌")

        with st.expander(f"{icon} [{row['优先级']}] {str(row['客诉文本'])[:70]}..."):
            ca, cb = st.columns([2, 1])
            with ca:
                st.markdown("**客诉原文**")
                st.text(row["客诉文本"])
                if "LLM分类理由" in row and pd.notna(row.get("LLM分类理由")):
                    st.caption(f"🤖 AI 分类理由: {row['LLM分类理由']}")
                st.markdown("**处理建议**")
                if "AI处理建议" in row and pd.notna(row.get("AI处理建议")):
                    st.success(row["AI处理建议"])
                else:
                    st.info(row.get("处理建议", ""))
            with cb:
                st.markdown(f"**分类**: {row['分类结果']}（置信度 {row['置信度']:.0%}）")
                st.markdown(f"**优先级**: :{pc.get(row['优先级'], 'green')}[{row['优先级']}]")
                st.markdown(f"**情绪**: {row['情绪']}")
                if "是否复合" in row and row["是否复合"]:
                    st.markdown("**⚠️ 复合投诉**")



def show_analytics(df):
    st.subheader("统计分析看板")

    c1, c2 = st.columns(2)
    with c1:
        cat_c = df["分类结果"].value_counts()
        fig = px.pie(values=cat_c.values, names=cat_c.index, title="客诉分类分布",
                     color_discrete_sequence=px.colors.qualitative.Set2, hole=0.4)
        fig.update_traces(textinfo="label+percent+value")
        st.plotly_chart(fig, width='stretch')
    with c2:
        pri_c = df["优先级"].value_counts()
        cmap = {"P0-紧急": "#FF4444", "P1-重要": "#FFA726", "P2-普通": "#66BB6A"}
        fig = px.bar(x=pri_c.index, y=pri_c.values, title="优先级分布",
                     color=pri_c.index, color_discrete_map=cmap, labels={"x": "优先级", "y": "数量"})
        st.plotly_chart(fig, width='stretch')

    c3, c4 = st.columns(2)
    with c3:
        sent_c = df["情绪"].value_counts()
        fig = px.bar(x=sent_c.index, y=sent_c.values, title="情绪分布",
                     color=sent_c.index, color_discrete_map={"愤怒": "#FF4444", "焦虑": "#FFA726", "平静": "#66BB6A"})
        st.plotly_chart(fig, width='stretch')
    with c4:
        cross = pd.crosstab(df["分类结果"], df["优先级"])
        fig = px.imshow(cross.values, x=cross.columns, y=cross.index,
                        title="分类×优先级交叉分析", color_continuous_scale="Reds", text_auto=True)
        st.plotly_chart(fig, width='stretch')

    if "create_time" in df.columns:
        st.subheader("时间趋势")
        try:
            dft = df.copy()
            dft["create_time"] = pd.to_datetime(dft["create_time"])
            dft["日期"] = dft["create_time"].dt.date
            td = dft.groupby(["日期", "分类结果"]).size().reset_index(name="数量")
            fig = px.line(td, x="日期", y="数量", color="分类结果", title="客诉日趋势", markers=True)
            st.plotly_chart(fig, width='stretch')
        except Exception:
            st.caption("时间字段无法解析")


def show_anomaly(df, model_key, client):
    st.subheader("🔍 批量异常检测")

    config = MODELS[model_key]

    # 关键词引擎
    st.markdown("#### 🔧 关键词规则引擎")
    kw = keyword_anomaly_detection(df, min_count=3)
    if kw:
        st.warning(f"检测到 **{len(kw)}** 个疑似批量异常")
        for a in kw:
            with st.expander(f"{a['预警等级']} {a['异常主题']} — 影响 {a['影响单量']} 单", expanded=False):
                st.markdown(f"**关联关键词**: {a['关联关键词']}")
                st.markdown(f"**影响面**: {a['影响单量']} 单")
                st.info("1. 定位涉事商品/商家/物流商\n2. 核实影响面\n3. 制定批量处理策略\n4. 输出话术通知一线")
    else:
        st.success("未检测到关键词级批量异常")

    # AI 引擎
    if client:
        st.divider()
        st.markdown(f"#### 🤖 {config['name']} 语义聚类")

        with st.spinner(f"{config['name']} 正在进行语义级异常聚类..."):
            llm_a = llm_batch_anomaly(df, model_key, client)

        if llm_a:
            st.warning(f"AI 语义聚类检测到 **{len(llm_a)}** 个异常事件")
            for a in llm_a:
                sev = a.get("severity", "🟡 黄色预警")
                with st.expander(f"{sev} {a.get('topic', '未命名')} — 约 {a.get('affected_count', '?')} 单", expanded=True):
                    x1, x2 = st.columns(2)
                    with x1:
                        st.markdown(f"**描述**: {a.get('description', '')}")
                        st.markdown(f"**关键词**: {'、'.join(a.get('keywords', []))}")
                        st.markdown(f"**影响**: 约 {a.get('affected_count', '?')} 单")
                    with x2:
                        st.markdown("**建议动作**")
                        st.info(a.get("recommended_action", "核实后制定应对策略"))
        else:
            st.success("AI 语义聚类未发现批量异常事件")
    else:
        st.info(f"💡 {config['name']} 未连接，输入 API Key 或启动 Ollama 后可体验 AI 异常检测")


def show_export(df):
    st.subheader("📥 导出分析结果")
    e1, e2 = st.columns([1, 3])
    with e1:
        csv = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("⬇️ 下载完整分析结果 (CSV)", data=csv,
                           file_name=f"客诉分类结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv", width='stretch')


def _render_engine_card(category, confidence, sentiment, priority, suggestion, extra=None):
    """渲染单个引擎结果卡片"""
    icon = CATEGORY_RULES.get(category, {}).get("icon", "📌")
    pc = {"P0-紧急": "red", "P1-重要": "orange", "P2-普通": "green"}
    sc = {"愤怒": "red", "焦虑": "orange", "平静": "blue"}

    st.markdown(f"**{icon} 分类**: {category}（置信度 {confidence:.0%}）")
    st.markdown(f"**优先级**: :{pc.get(priority, 'green')}[{priority}]")
    st.markdown(f"**情绪**: :{sc.get(sentiment, 'blue')}[{sentiment}]")
    if suggestion:
        st.markdown(f"**处理建议**: {suggestion}")
    if extra:
        for label, value in extra.items():
            if value:
                st.caption(f"*{label}: {value}*")


def generate_keyword_results(df):
    """对 DataFrame 执行批量关键词分析"""
    progress_bar = st.progress(0)
    total = len(df)
    categories, confidences, sentiments, priorities, suggestions = [], [], [], [], []

    for i, (_, row) in enumerate(df.iterrows()):
        text = row["客诉文本"]
        cat, conf = keyword_classify(text)
        sent = keyword_sentiment(text)
        amount = None
        if "order_amount" in df.columns:
            try:
                amount = float(row["order_amount"]) if pd.notna(row.get("order_amount")) else None
            except (ValueError, TypeError):
                amount = None
        pri = keyword_priority(text, sent, amount)
        sug = keyword_suggestion(cat, sent, pri)

        categories.append(cat)
        confidences.append(conf)
        sentiments.append(sent)
        priorities.append(pri)
        suggestions.append(sug)
        progress_bar.progress((i + 1) / total)

    progress_bar.empty()
    return categories, confidences, sentiments, priorities, suggestions


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════

def main():
    init_session()

    # 标题栏
    c1, c2 = st.columns([3, 1])
    with c1:
        st.title("🔍 客诉智能分类系统")
        st.caption("多模型 AI 引擎 | 关键词规则 + Ollama + DeepSeek + Gemini + Groq")
    with c2:
        st.metric("版本", "v3.2", delta="分析深度增强")
        ollama_ok = check_ollama_available() and check_ollama_model()
        st.metric("本地 AI", "就绪" if ollama_ok else "待启动")

    st.divider()

    # 侧边栏
    selected_model = show_sidebar()
    config = MODELS[selected_model]
    is_rule_only = (config["sdk_type"] == "rule")

    # 主区域
    if st.session_state.get("working_df") is None:
        show_welcome()
    else:
        df = st.session_state["working_df"].copy()
        if "客诉文本" not in df.columns and "complaint_text" in df.columns:
            df["客诉文本"] = df["complaint_text"]

        # 关键词引擎（始终运行）
        with st.spinner("🔧 关键词规则引擎分析中..."):
            kw_cats, kw_confs, kw_sents, kw_pris, kw_sugs = generate_keyword_results(df)
            df["分类结果"] = kw_cats
            df["置信度"] = kw_confs
            df["情绪"] = kw_sents
            df["优先级"] = kw_pris
            df["处理建议"] = kw_sugs

        # AI 引擎 — 仅非规则模式且客户端就绪时运行
        client = None if is_rule_only else get_client(selected_model)

        if is_rule_only:
            st.success(f"🔧 使用**关键词规则引擎**完成分析，共 {len(df)} 条客诉。切换到 AI 引擎可获得语义级分析。")
        elif client:
            st.info(f"🤖 启动 {config['name']} 语义分析...（处理 80 条客诉约需 1-3 分钟）")

            ai_cats, ai_confs, ai_sents, ai_pris, ai_sugs = [], [], [], [], []
            ai_reasonings, ai_compounds = [], []

            progress_bar = st.progress(0)
            total = len(df)

            for i, (_, row) in enumerate(df.iterrows()):
                result = llm_classify(row["客诉文本"], selected_model, client)
                if result:
                    ai_cats.append(result.get("category", "其他"))
                    ai_confs.append(result.get("confidence", 0.0))
                    ai_sents.append(result.get("sentiment", "平静"))
                    ai_pris.append(result.get("priority", "P2-普通"))
                    ai_sugs.append(result.get("action_recommendation", ""))
                    ai_reasonings.append(result.get("reasoning", ""))
                    ai_compounds.append(result.get("is_compound", False))
                else:
                    ai_cats.append("其他"); ai_confs.append(0.0); ai_sents.append("平静")
                    ai_pris.append("P2-普通"); ai_sugs.append(""); ai_reasonings.append("")
                    ai_compounds.append(False)
                progress_bar.progress((i + 1) / total)

            progress_bar.empty()

            df["AI分类结果"] = ai_cats
            df["AI置信度"] = ai_confs
            df["AI情绪"] = ai_sents
            df["AI优先级"] = ai_pris
            df["AI处理建议"] = ai_sugs
            df["LLM分类理由"] = ai_reasonings
            df["是否复合"] = ai_compounds

            st.success(f"✅ {config['name']} 分析完成，共处理 {total} 条客诉")
        else:
            # AI 未就绪——不报错，展示规则结果并引导配置
            if config["key_required"]:
                st.info(f"💡 **{config['name']}** 需要 API Key 才能启用。当前展示的是**关键词规则引擎**结果。请在侧边栏输入 Key 或切换到「关键词规则引擎」模式。")
            elif selected_model == "ollama-qwen":
                ollama_running = check_ollama_available()
                if not ollama_running:
                    st.warning("⚠️ Ollama 未运行。当前展示**关键词规则引擎**结果。安装 Ollama 后可免费使用本地 AI。")
                    st.code("下载 Ollama: https://ollama.com/download/windows\n安装后运行: ollama pull qwen2.5:3b", language="bash")
                else:
                    st.warning("⚠️ Qwen2.5 模型未拉取。当前展示**关键词规则引擎**结果。运行 `ollama pull qwen2.5:3b` 拉取模型后即可使用。")

        # 展示结果
        show_batch_analysis(df, selected_model, client if not is_rule_only else None)


if __name__ == "__main__":
    main()
