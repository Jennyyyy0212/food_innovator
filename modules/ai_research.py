import json
import re
import time
from pathlib import Path
from typing import Callable, Optional
import yaml

from gemini_utils import gemini_generate as _gemini_generate, parse_json_loose

import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from google.genai.types import File

from modules.tw_additive_rag import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_VECTOR_DIR,
    get_tw_source,
    load_tw_vector_store,
    retrieve_tw_additive_context,
    tw_reference_caption,
    tw_staleness_warning_message,
    vector_store_dir_ready,
    retrieve_tw_additive_context_exact_first,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = _PROJECT_ROOT / "uploaded_reg_files.json"

COUNTRY_CODE_TO_NAME = {
    "tw": "台灣",
    "jp": "日本",
    "us": "美國",
    "hk": "香港",
    "mo": "澳門",
    "cn": "中國",
    "kr": "韓國",
    "sg": "新加坡",
    "th": "泰國",
    "vn": "越南",
}
COUNTRY_NAME_TO_CODE = {v: k for k, v in COUNTRY_CODE_TO_NAME.items()}

# -----------------------------------------------------------
# Country RAG registry
#
# To add a new country, append an entry here.  Required keys:
#   code          – ISO 2-letter country code
#   vector_dir    – Path to the FAISS index directory
#   manifest_path – Path to the sources_manifest.yaml for this country
#   source_getter – callable(manifest_path=...) -> dict | None
#   retriever     – callable(vector_store, items: list[str], k: int) -> str
#   ready_checker – callable(vector_dir) -> bool
#   cache_label   – human-readable label for st.cache_resource spinner
# -----------------------------------------------------------
COUNTRY_CONFIGS = {
    "tw": {
        "code": "tw",
        "vector_dir": DEFAULT_VECTOR_DIR,
        "manifest_path": DEFAULT_MANIFEST_PATH,
        "source_getter": get_tw_source,
        "retriever": lambda vs, items, k: retrieve_tw_additive_context_exact_first(vs, items, k=k),
        "ready_checker": vector_store_dir_ready,
        "cache_label": "台灣添加物向量索引",
    },
    # Example – uncomment and fill in when Japan RAG is ready:
    # "jp": {
    #     "code": "jp",
    #     "vector_dir": _PROJECT_ROOT / "data" / "processed" / "japan" / "vector_store",
    #     "manifest_path": _PROJECT_ROOT / "data" / "japan_manifest.yaml",
    #     "source_getter": get_jp_source,
    #     "retriever": lambda vs, items, k: retrieve_tw_additive_context(vs, items, k=k),
    #     "ready_checker": vector_store_dir_ready,
    #     "cache_label": "日本添加物向量索引",
    # },
}

_COUNTRY_FLAGS = {
    "tw": "🇹🇼", "jp": "🇯🇵", "us": "🇺🇸", "hk": "🇭🇰",
    "mo": "🇲🇴", "cn": "🇨🇳", "kr": "🇰🇷", "sg": "🇸🇬",
    "th": "🇹🇭", "vn": "🇻🇳",
}

OFFICIAL_SOURCE_HINTS = {
    "jp": {
        "name": "Japan MHLW Food Additives",
        "url": "https://www.mhlw.go.jp/"
    },
    "us": {
        "name": "US FDA Food Additives",
        "url": "https://www.fda.gov/food/food-additives-petitions"
    },
    "sg": {
        "name": "Singapore Food Agency Food Additives",
        "url": "https://www.sfa.gov.sg/"
    },
}


@st.cache_resource(show_spinner="載入法規向量索引…")
def _cached_country_faiss(vector_dir_str: str, google_api_key: str, embedding_model: str):
    return load_tw_vector_store(
        google_api_key=google_api_key,
        vector_dir=Path(vector_dir_str),
        embedding_model=embedding_model,
    )


# -----------------------------------------------------------
# File utilities
# -----------------------------------------------------------

def clean_html(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    for a in soup.find_all("a", href=True):
        link_text = a.get_text(strip=True)
        href = a["href"].strip()
        a.replace_with(f"{link_text} ({href})" if link_text else href)
    return soup.get_text(separator="\n", strip=True)


def ensure_clean_txt_files(
    folder_path="html_pages",
    force_clean=False,
    log_fn: Optional[Callable[[str], None]] = None,
):
    html_dir = Path(folder_path)
    txt_files = []
    for html_file in html_dir.glob("*.html"):
        txt_path = html_file.with_suffix(".txt")
        if force_clean or not txt_path.exists():
            html = html_file.read_text(encoding="utf-8")
            txt_path.write_text(clean_html(html), encoding="utf-8")
            if log_fn:
                log_fn(f"✅ Cleaned {html_file.name}")
        else:
            if log_fn:
                log_fn(f"🔁 Using cached: {txt_path.name}")
        txt_files.append(txt_path)
    return txt_files


def is_file_expired(info, hours=23):
    return time.time() - info.get("uploaded_at", 0) > hours * 3600


def upload_and_cache_files(
    client,
    txt_files,
    force=False,
    log_fn: Optional[Callable[[str], None]] = None,
):
    uploaded_cache = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    for txt in txt_files:
        name = txt.name
        file_info = uploaded_cache.get(name)
        if force or (not file_info) or is_file_expired(file_info):
            if log_fn:
                log_fn(f"⬆️ Uploading {name}…")
            file = client.files.upload(file=txt)
            uploaded_cache[name] = {
                "file_uri": file.uri,
                "file_name": file.name,
                "uploaded_at": time.time(),
            }
            if log_fn:
                log_fn(f"☁️ Uploaded {name}")
        else:
            if log_fn:
                log_fn(f"✅ 使用快取: {name}")
    STATE_FILE.write_text(json.dumps(uploaded_cache, indent=2, ensure_ascii=False))
    return uploaded_cache


def search_official_regulation_source(country_code: str):
    return OFFICIAL_SOURCE_HINTS.get(country_code)


def _clean_reg_item_name(item: str) -> str:
    s = re.sub(r"[（(].*?[）)]", "", str(item).strip())
    return re.split(r"\s*[-－—:：]\s*", s)[0].strip()


def _dedupe_items(seq):
    seen = set()
    out = []
    for x in seq:
        s = str(x).strip() if x is not None else ""
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def country_display_name(country_code: str) -> str:
    return COUNTRY_CODE_TO_NAME.get(country_code, country_code)


# -----------------------------------------------------------
# Main render
# -----------------------------------------------------------

def render_research(client, model_name):
    st.header("🔬 深度研發與法規 AI")
    st.caption("透過 RAG 向量索引結合 Gemini，查詢各地食品添加物法規合規狀態。")

    process_log = st.sidebar.empty()

    def log_msg(msg):
        if "log_lines" not in st.session_state:
            st.session_state.log_lines = []
        st.session_state.log_lines.append(msg)
        process_log.text("\n".join(st.session_state.log_lines[-5:]))

    def gemini_generate(prompt) -> str:
        return _gemini_generate(
            client,
            model_name,
            prompt,
            on_retry=lambda attempt, delay: st.toast(
                f"⏳ Gemini 暫時繁忙，{delay} 秒後重試（第 {attempt + 1} 次）…"
            ),
            on_error=lambda e: st.error(f"❌ Gemini 錯誤：{e}"),
        )

    # -----------------------------------------------------------
    # Description
    # -----------------------------------------------------------
    st.markdown("""
本工具透過 **RAG（檢索增強生成）** 技術，結合本地向量索引與 Gemini，分析食品法規與添加物合規狀態。

**目前支援狀況：**
| 地區 | 狀態 | 說明 |
|------|------|------|
| 🇹🇼 台灣 | ✅ 已整合 | FAISS 本地向量索引（需先執行 `python scripts/build_tw_chunks.py --embed`） |
| 🇭🇰 香港 | 🔧 整合中 | 已收集官方法規 HTML，尚未建立向量索引 |
| 🇲🇴 澳門 | 🔧 整合中 | 已收集官方法規 HTML，尚未建立向量索引 |
| 🌏 其他地區 | ⏳ 計劃中 | 尚未建立索引，查詢結果將標示「資料不足」 |
""")

    st.markdown("### 📚 法規資料來源")
    st.markdown("""
**已整合（可查詢）：**
- 🇹🇼 [台灣食品添加物法規](https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=L0040001) ✅

**已收集、整合中：**
- 🇭🇰 [香港食物安全中心 - 食品添加劑守則](https://www.cfs.gov.hk/tc_chi/food_leg/food_leg.html) 🔧
- 🇲🇴 [澳門食品安全中心 - 食品添加劑資料](https://www.iam.gov.mo/foodsafety/c/lawstandard/list) 🔧

**計劃中（尚未建立索引）：**
- 🇯🇵 [日本食品衛生法 (MHLW)](https://www.mhlw.go.jp/stf/seisakunitsuite/bunya/0000066597.html)
- 🇨🇳 [中國食品添加劑使用標準 GB 2760](https://www.samr.gov.cn/)
- 🇰🇷 [韓國食品添加物法規 (MFDS)](https://www.mfds.go.kr/)
- 🇸🇬 [新加坡食品法規 (SFA)](https://www.sfa.gov.sg/food-information/food-labelling/food-additives)
- 🇺🇸 [US FDA Food Ingredients Regulations](https://www.fda.gov/food/food-additives-petitions/food-additive-status-list)
- 🇪🇺 [EU Food Additives Database](https://webgate.ec.europa.eu/foods_system/main/?sector=FAD)
""")

    all_asian_countries = ["台灣", "日本", "中國", "韓國", "新加坡", "泰國", "越南", "香港", "澳門"]
    selected_countries = st.multiselect(
        "🌏 選擇要查詢的地區（可多選）",
        options=all_asian_countries,
        default=["台灣"],
    )

    if "台灣" in selected_countries:
        tw_src = get_tw_source(manifest_path=DEFAULT_MANIFEST_PATH)
        ref = tw_reference_caption(tw_src) if tw_src else ""
        official = (tw_src or {}).get("official_url") or ""
        cap_parts = ["台灣食品添加物索引（本地資料）"]
        if ref:
            cap_parts.append(ref)
        if official:
            cap_parts.append(f"官方列表：{official}")
        st.caption(" ".join(cap_parts))
        stale_msg = tw_staleness_warning_message(tw_src) if tw_src else None
        if stale_msg:
            st.warning(stale_msg)
        if not vector_store_dir_ready(DEFAULT_VECTOR_DIR):
            st.info(
                "尚未偵測到台灣向量索引。請於專案根目錄執行 "
                "`python scripts/build_tw_chunks.py --embed` 並設定 GOOGLE_API_KEY 後重啟。"
                "在索引建立前，台灣查詢將返回「資料不足」。"
            )

    with st.sidebar:
        if st.button("🔄 重新上傳 HK/MO 法規參考資料"):
            txt_files = ensure_clean_txt_files("html_pages", force_clean=True, log_fn=log_msg)
            if STATE_FILE.exists():
                STATE_FILE.unlink()
            upload_and_cache_files(client, txt_files, force=True, log_fn=log_msg)
            st.success("✅ 法規參考資料已重新上傳到 Gemini。")

    st.divider()
    st.subheader("🌎 法規與食材分析查詢")
    st.markdown("輸入或貼上要分析的食品概念或配方 JSON，AI 將查詢各地添加物合規狀態。")

    # Country status summary — shows what's selected and its support level
    if selected_countries:
        badges = []
        for country in selected_countries:
            code = COUNTRY_NAME_TO_CODE.get(country, "")
            flag = _COUNTRY_FLAGS.get(code, "🌏")
            tag  = "✅ RAG" if code in COUNTRY_CONFIGS else "⏳ 資料不足"
            badges.append(f"{flag} {country} ({tag})")
        st.caption("查詢地區：" + "　·　".join(badges))

    if "concept_input" not in st.session_state:
        st.session_state.concept_input = ""

    concept_input = st.text_area(
        "🧾 想分析的概念 / 食品說明",
        value=st.session_state.concept_input,
        placeholder="例如：抹茶流心大福 / 紫薯拿鐵 / 便利店甜點組合包",
    )

    if st.button("🔍 分析食譜與各地法規", use_container_width=True):
        if not concept_input.strip():
            st.warning("請先輸入要分析的食譜概念。")
        else:
            with st.status("🔍 AI 正在分析法規...", expanded=True) as _reg_status:
                extract_prompt = f"""
                你是食品法規資料抽取器。

                請從以下既有配方 JSON 中，只抽取「食品添加物」。
                不要抽取一般食材，例如奶油乳酪、雞蛋、糖、麵粉、鮮奶油、抹茶粉。

                配方 JSON：
                {json.dumps(concept_input, ensure_ascii=False, indent=2)}

                請輸出有效 JSON：
                {{
                "regulatory_items": [
                    {{
                    "name": "添加物中文名稱",
                    "english_name": "英文名稱或空字串",
                    "amount": "用量，例如 0.5%",
                    "function": "用途，例如 防腐、增稠、酸度調節"
                    }}
                ]
                }}

                只輸出 JSON，不要 Markdown。
                """

                extract_json = parse_json_loose(gemini_generate(extract_prompt))
                reg_items = extract_json.get("regulatory_items", [])

                items = _dedupe_items([
                    _clean_reg_item_name(x.get("name", ""))
                    for x in reg_items
                ])

                if not items:
                    st.info("未偵測到特定添加物，AI 將僅顯示主要食材資訊。")

                example_block = """
                        [
                        {
                            "國家": "台灣",
                            "項目": "山梨酸鉀",
                            "類型": "食品添加物",
                            "使用狀態": "允許",
                            "最大添加量": "0.6 g/kg",
                            "適用食品類別": "飲料、糕點",
                            "標示或衛生要求": "須標示防腐劑名稱",
                            "條文或來源": "食品添加物法規 第 15 條"
                        }
                        ]
                """

                def _batch_country_rag_prompt(
                    country: str,
                    item_list: list,
                    rag_text: str,
                    source: dict | None,
                ) -> str:
                    payload = json.dumps(item_list, ensure_ascii=False)
                    as_of = (source or {}).get("as_of_date") or ""
                    official_url = (source or {}).get("official_url") or ""

                    return f"""
                你是一位量產食品法規專家。
                以下摘錄來自「{country}」食品法規資料庫。
                資料基準日：{as_of}
                官方參考：{official_url}

                【摘錄（檢索結果）】
                {rag_text}

                ---
                請**僅依摘錄**分析「{country}」地區下列每一個項目。
                摘錄未載明處請標「資料不足」，不可自行推測。

                項目列表（JSON）：{payload}

                請輸出有效 JSON 陣列，每個物件包含：
                - 國家
                - 項目
                - 類型
                - 使用狀態：允許 / 禁止 / 資料不足
                - 最大添加量
                - 適用食品類別
                - 標示或衛生要求
                - 條文或來源：必須使用摘錄中的 official_url；若有 item_no，請一起列出

                每個項目都必須有一個 JSON 物件。
                不要包含 Markdown 或額外文字。
                """

                def _append_reg_parse(reg_json, raw_text: str, country: str, batch_label: str):
                    if isinstance(reg_json, list):
                        reg_results.extend(reg_json)
                    elif isinstance(reg_json, dict):
                        reg_results.append(reg_json)
                    else:
                        unrecognized_outputs.append({
                            "country": country,
                            "item": batch_label,
                            "raw_text": raw_text or "(空輸出)",
                        })

                reg_results = []
                unrecognized_outputs = []

                rag_countries = [
                    COUNTRY_NAME_TO_CODE[c]
                    for c in selected_countries
                    if COUNTRY_NAME_TO_CODE.get(c) in COUNTRY_CONFIGS
                ]
                missing_country_codes = [
                    COUNTRY_NAME_TO_CODE[c]
                    for c in selected_countries
                    if COUNTRY_NAME_TO_CODE.get(c) not in COUNTRY_CONFIGS
                ]

                if not items:
                    st.info("沒有可查詢的食材或添加物項目，已跳過法規批次查詢。")
                else:
                    # 1. RAG countries (have a vector index in COUNTRY_CONFIGS)
                    for country_code in rag_countries:
                        _flag = _COUNTRY_FLAGS.get(country_code, "🌏")
                        _reg_status.write(f"{_flag} 查詢 {country_display_name(country_code)}…")
                        cfg = COUNTRY_CONFIGS[country_code]
                        source = cfg["source_getter"](manifest_path=cfg["manifest_path"])
                        api_key = (st.session_state.get("api_key_input") or "").strip()

                        use_rag = bool(api_key) and cfg["ready_checker"](cfg["vector_dir"])
                        rag_text = ""

                        log_msg(f"[{country_code}] API key: {bool(api_key)}, index ready: {cfg['ready_checker'](cfg['vector_dir'])}")

                        if use_rag:
                            try:
                                vs = _cached_country_faiss(
                                    str(cfg["vector_dir"]),
                                    api_key,
                                    DEFAULT_EMBEDDING_MODEL,
                                )
                                if vs is not None:
                                    rag_text = cfg["retriever"](vs, items, 6)
                                    log_msg(f"[{country_code}] Retrieved {len(rag_text)} chars")
                            except Exception as e:
                                log_msg(f"[{country_code}] 向量索引載入失敗：{e}")
                                use_rag = False

                        if use_rag and rag_text.strip():
                            log_msg(f"[{country_code}] 使用 RAG 檢索分析")
                            prompt = _batch_country_rag_prompt(
                                country_display_name(country_code), items, rag_text, source
                            )
                            text = gemini_generate(prompt)
                            reg_json = parse_json_loose(text)
                            _append_reg_parse(reg_json, text, country_code, f"批次：{len(items)} 項（RAG）")
                        else:
                            for item in items:
                                reg_results.append({
                                    "國家": country_display_name(country_code),
                                    "項目": item,
                                    "使用狀態": "資料不足",
                                    "原因": "RAG 未執行：可能是 API key 缺失、索引不存在，或未檢索到相關資料",
                                })

                    # 2. Countries without a vector index yet
                    for country_code in missing_country_codes:
                        _flag = _COUNTRY_FLAGS.get(country_code, "🌏")
                        _reg_status.write(f"{_flag} {country_display_name(country_code)} 尚無向量索引…")
                        source = search_official_regulation_source(country_code)
                        if not source:
                            for item in items:
                                reg_results.append({
                                    "國家": country_display_name(country_code),
                                    "項目": item,
                                    "使用狀態": "資料不足",
                                    "原因": "尚未建立該國 RAG 索引，也未找到可信官方來源",
                                })
                            continue
                        reg_results.append({
                            "國家": country_display_name(country_code),
                            "項目": "資料來源",
                            "使用狀態": "需要建立索引",
                            "條文或來源": source["url"],
                            "原因": "已找到官方來源，但尚未轉成 RAG 向量索引",
                        })

                _reg_status.update(label="✅ 法規查詢完成", state="complete")

            try:
                _no_data = {"資料不足", "需要建立索引"}
                displayable = [r for r in reg_results if r.get("使用狀態") not in _no_data]
                skipped_countries = {
                    r.get("國家") for r in reg_results if r.get("使用狀態") in _no_data
                } - {r.get("國家") for r in displayable}

                if displayable:
                    df = pd.DataFrame(displayable)
                    cols = ["國家", "項目", "類型", "使用狀態", "最大添加量", "適用食品類別", "標示或衛生要求", "條文或來源"]
                    df = df[[c for c in cols if c in df.columns]]
                    st.dataframe(df, use_container_width=True)
                    countries_with_data = sorted({r.get("國家") for r in displayable})
                    st.success("✅ 完成法規查詢（RAG 向量檢索）：" + "、".join(countries_with_data))
                else:
                    st.warning("⚠️ 所有選擇地區均無可用法規資料，請先建立對應的 RAG 向量索引。")

                if skipped_countries:
                    st.caption("⏳ 資料不足，已略過：" + "、".join(sorted(skipped_countries)))

                if unrecognized_outputs:
                    st.warning("⚠️ 以下項目未能解析為有效 JSON：")
                    for bad in unrecognized_outputs:
                        st.markdown(f"**{bad['country']} - {bad['item']}**")
                        st.code(bad["raw_text"], language="json")
            except Exception as e:
                st.error(f"無法顯示法規查詢結果：{e}")
