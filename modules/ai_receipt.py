import json
import requests
import streamlit as st

from gemini_utils import gemini_generate as _gemini_generate, parse_json_loose
from modules.ai_innovation import _load_favorites, _save_favorites


_OFF_API = "https://world.openfoodfacts.org/cgi/search.pl"


# -----------------------------------------------------------
# Recipe markdown formatter
# -----------------------------------------------------------

def _fmt_pct(v) -> str:
    try:
        return f"{float(v):.1f}%"
    except (TypeError, ValueError):
        return str(v)


def recipe_to_markdown(r: dict) -> str:
    """Convert a receipt JSON dict to a human-readable Markdown string."""
    lines = []
    product_name = r.get("product_name", "配方")
    concept      = r.get("product_concept", "")
    total        = r.get("total_weight_g", 1000)

    lines.append(f"# 🧾 {product_name}")
    if concept:
        lines.append(f"\n> {concept}\n")
    lines.append(f"**總重量：** {total} g\n")

    ingredients = r.get("ingredients", [])
    if ingredients:
        lines.append("## 🥘 食材\n")
        lines.append("| 食材 | 用量 (g) | 比例 | 功能 |")
        lines.append("|------|:--------:|------|------|")
        for ing in ingredients:
            lines.append(
                f"| {ing.get('name','')} | {ing.get('weight_g',0)} "
                f"| {_fmt_pct(ing.get('percentage',0))} | {ing.get('function','')} |"
            )
        lines.append("")

    additives = r.get("additives", [])
    if additives:
        lines.append("## 🧪 添加物\n")
        lines.append("| 添加物 | 用量 (g) | 比例 | 用途 |")
        lines.append("|--------|:--------:|------|------|")
        for add in additives:
            lines.append(
                f"| {add.get('name','')} | {add.get('weight_g',0)} "
                f"| {_fmt_pct(add.get('percentage',0))} | {add.get('purpose','')} |"
            )
        lines.append("")

    process = r.get("process", [])
    if process:
        lines.append("## 🔧 製程步驟\n")
        for i, step in enumerate(process, 1):
            lines.append(f"{i}. {step}")
        lines.append("")

    notes = r.get("mass_production_notes", [])
    if notes:
        lines.append("## 📦 量產備注\n")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    reg = r.get("regulatory_check_items", [])
    if reg:
        lines.append("## ⚖️ 法規查核項目\n")
        for item in reg:
            lines.append(f"- {item}")

    return "\n".join(lines)

# -----------------------------------------------------------
# Open Food Facts helpers
# -----------------------------------------------------------

def _search_open_food_facts(query: str, page_size: int = 5) -> list:
    try:
        resp = requests.get(
            _OFF_API,
            params={
                "search_terms": query,
                "search_simple": 1,
                "action": "process",
                "json": 1,
                "page_size": page_size,
                "fields": "product_name,ingredients_text,additives_tags",
            },
            timeout=10,
        )
        resp.raise_for_status()
        return [p for p in resp.json().get("products", []) if p.get("ingredients_text", "").strip()]
    except Exception:
        return []


def _format_off_for_prompt(products: list) -> str:
    if not products:
        return ""
    lines = ["【真實產品參考配方（來源：Open Food Facts）】"]
    for i, p in enumerate(products, 1):
        name = p.get("product_name") or "Unknown"
        ingredients = (p.get("ingredients_text") or "").strip()
        additives = p.get("additives_tags") or []
        lines.append(f"\n產品 {i}：{name}")
        lines.append(f"  食材：{ingredients[:400]}{'…' if len(ingredients) > 400 else ''}")
        if additives:
            lines.append(f"  添加物代碼：{', '.join(additives)}")
    return "\n".join(lines)


# -----------------------------------------------------------
# Main render
# -----------------------------------------------------------

def render_receipt(client, model_name):
    st.header("🧾 Receipt Generator")
    st.caption("將產品概念轉成 JSON 配方：食材重量、添加物、比例與量產建議。")

    # Session state
    for key, default in [
        ("receipt_json", {}),
        ("receipt_whitelist", []),
        ("receipt_whitelist_concept", ""),
        ("receipt_off_products", []),
        ("receipt_off_keywords", ""),
        ("receipt_off_tried", []),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    def gemini_generate(prompt: str) -> str:
        return _gemini_generate(
            client, model_name, prompt,
            on_retry=lambda attempt, delay: st.toast(
                f"⏳ Gemini 暫時繁忙，{delay} 秒後重試（第 {attempt + 1} 次）…"
            ),
            on_error=lambda e: st.error(f"❌ Gemini 錯誤：{e}"),
        )

    # -----------------------------------------------------------
    # Inputs
    # -----------------------------------------------------------
    # Consume any pre-fill staged from another tab BEFORE the widget is created.
    if "receipt_concept_prefill" in st.session_state:
        st.session_state["receipt_concept_input"] = st.session_state.pop("receipt_concept_prefill")

    concept = st.text_area(
        "產品概念",
        placeholder="例如：便利店販售的抹茶流心大福，冷藏保存，主打年輕女性與下午茶場景。",
        key="receipt_concept_input",
    )

    use_off = st.toggle(
        "🔍 參考真實產品配方（Open Food Facts）",
        value=False,
        help="開啟後分析概念時會自動提取英文關鍵字並搜尋 Open Food Facts，結果注入配方提示詞。",
    )

    # Warn if concept changed since last analysis
    concept_changed = (
        concept.strip()
        and st.session_state.receipt_whitelist
        and concept.strip() != st.session_state.receipt_whitelist_concept
    )
    if concept_changed:
        st.warning("⚠️ 概念已更改，建議重新分析食材清單。")

    # -----------------------------------------------------------
    # Phase 1 — Analyse concept: whitelist + OFF search
    # -----------------------------------------------------------
    if st.button("📋 分析概念與食材", use_container_width=True):
        if not concept.strip():
            st.warning("請先輸入產品概念。")
            st.stop()

        # 1a. Generate relevant ingredient whitelist for this specific concept
        with st.spinner("AI 正在分析相關食材…"):
            whitelist_raw = gemini_generate(f"""
你是食品配方專家。請為以下產品概念，列出 12–18 個最相關的食材與食品添加物，
作為配方設計的參考白名單。只列出真實且常見的品項，不要臆測或編造。

產品概念：{concept.strip()}

請以 JSON 字串陣列格式輸出，例如：["水", "砂糖", "抹茶粉", "山梨酸鉀"]
只輸出 JSON 陣列，不要其他說明。
""")
        parsed = parse_json_loose(whitelist_raw)
        if isinstance(parsed, list) and parsed:
            st.session_state.receipt_whitelist = [str(x).strip() for x in parsed if x]
        else:
            # fallback: split by comma if Gemini returned plain text
            st.session_state.receipt_whitelist = [
                x.strip().strip('"') for x in whitelist_raw.replace("、", ",").split(",")
                if x.strip()
            ]
        st.session_state.receipt_whitelist_concept = concept.strip()

        # 1b. OFF search — Gemini generates 3-5 keyword variations,
        #     try each from specific → broad until results are found.
        if use_off:
            with st.spinner("AI 正在構思搜尋關鍵字…"):
                kw_raw = gemini_generate(f"""
You are helping search for real food products on Open Food Facts.
Generate 3–5 English search keyword variations for the concept below,
ordered from most specific to most broad (so we can fall back if specific terms return nothing).

Concept: "{concept.strip()}"

Think about:
- The exact product name in English
- Similar or related product types
- Broader category or flavour terms

Reply with one keyword phrase per line, no numbering, no explanation.
Example output:
matcha swiss roll cake
matcha roll cake
green tea roll cake
matcha cake
green tea cake
""")
            # Parse: one keyword per line, fall back to comma-split
            kw_lines = [
                ln.strip().strip('"').strip("'").strip("-").strip()
                for ln in kw_raw.strip().splitlines()
                if ln.strip() and not ln.strip()[0].isdigit()
            ]
            if not kw_lines:
                kw_lines = [k.strip() for k in kw_raw.split(",") if k.strip()]

            # Try each keyword in order, stop at first hit
            found_products: list = []
            found_keyword: str = ""
            tried: list[str] = []
            for kw in kw_lines:
                if not kw:
                    continue
                tried.append(kw)
                with st.spinner(f"搜尋 Open Food Facts：「{kw}」…"):
                    results = _search_open_food_facts(kw)
                if results:
                    found_products = results
                    found_keyword = kw
                    break

            st.session_state.receipt_off_products = found_products
            st.session_state.receipt_off_keywords = found_keyword
            st.session_state.receipt_off_tried = tried

            if not found_products:
                tried_str = "、".join(f"`{k}`" for k in tried)
                st.warning(f"Open Food Facts 嘗試了 {tried_str}，均未找到符合產品，配方將不含真實參考。")
        else:
            st.session_state.receipt_off_products = []
            st.session_state.receipt_off_keywords = ""
            st.session_state.receipt_off_tried = []

    # -----------------------------------------------------------
    # Phase 1 results display (shown after analysis, persists)
    # -----------------------------------------------------------
    if st.session_state.receipt_whitelist:
        st.markdown("---")
        st.markdown("#### 🥘 食材白名單（AI 根據概念生成，可調整）")

        # Show OFF search results if available
        if use_off and st.session_state.receipt_off_products:
            kw = st.session_state.receipt_off_keywords
            tried = st.session_state.get("receipt_off_tried", [])
            skipped = [k for k in tried if k != kw]
            products = st.session_state.receipt_off_products

            label = f"📦 找到 {len(products)} 個真實參考產品｜關鍵字：`{kw}`"
            if skipped:
                label += f"（已跳過：{', '.join(skipped)}）"

            with st.expander(label, expanded=True):
                for p in products:
                    name = p.get("product_name") or "Unknown"
                    ingredients = (p.get("ingredients_text") or "").strip()
                    st.markdown(f"**{name}**")
                    st.caption(ingredients[:280] + ("…" if len(ingredients) > 280 else ""))
        elif use_off and st.session_state.get("receipt_off_tried"):
            tried_str = "、".join(f"`{k}`" for k in st.session_state.receipt_off_tried)
            st.caption(f"已嘗試：{tried_str}｜無符合產品")

        # Editable whitelist multiselect
        all_opts = st.session_state.receipt_whitelist
        allowed_items = st.multiselect(
            "可新增或移除食材（全選為預設）",
            options=all_opts,
            default=all_opts,
            key="receipt_allowed_items",
        )

        # -----------------------------------------------------------
        # Phase 2 — Generate receipt
        # -----------------------------------------------------------
        if st.button("🧾 Generate Receipt", use_container_width=True):
            off_context = ""
            if use_off and st.session_state.receipt_off_products:
                off_context = _format_off_for_prompt(st.session_state.receipt_off_products)

            off_section = f"\n{off_context}\n" if off_context else ""
            prompt = f"""
你是一位食品配方計算專家。請根據產品概念，產生一份可用於初步研發的配方 JSON。

產品概念：
{concept.strip()}
{off_section}
可使用或優先參考的食材 / 添加物清單：
{json.dumps(allowed_items, ensure_ascii=False)}

規則：
1. 若有真實產品參考，請優先以其食材為依據，保持現實可行性。
2. 只使用清單內或清單外但明確真實存在的食材，不要編造原料。
3. 所有 ingredient 和 additive 都要有 weight_g。
4. total_weight_g 建議為 1000g。
5. percentage 必須依照 weight_g / total_weight_g 計算。
6. additives 必須說明 purpose。
7. 請輸出有效 JSON，不要 Markdown。

格式：
{{
  "product_name": "...",
  "product_concept": "...",
  "total_weight_g": 1000,
  "ingredients": [
    {{"name": "...", "weight_g": 0, "percentage": 0, "function": "..."}}
  ],
  "additives": [
    {{"name": "...", "weight_g": 0, "percentage": 0, "purpose": "..."}}
  ],
  "process": ["..."],
  "mass_production_notes": ["..."],
  "regulatory_check_items": ["..."]
}}
"""
            with st.spinner("Gemini 正在生成配方 JSON…"):
                text = gemini_generate(prompt)

            receipt_json = parse_json_loose(text)
            if not receipt_json:
                st.error("無法解析 Gemini 輸出，請重新生成。")
                st.code(text)
                st.stop()

            st.session_state.receipt_json = receipt_json
            st.session_state.concept_input = json.dumps(receipt_json, ensure_ascii=False, indent=2)

    # -----------------------------------------------------------
    # Result display
    # -----------------------------------------------------------
    if st.session_state.receipt_json:
        receipt = st.session_state.receipt_json
        st.markdown("---")

        # Markdown view
        st.markdown(recipe_to_markdown(receipt))

        st.markdown("---")

        # Action buttons
        col_fav, col_dl, col_reg = st.columns([1, 1, 1])

        with col_fav:
            if st.button("⭐ 加入收藏", use_container_width=True):
                favs = _load_favorites()
                entry = {
                    "type": "recipe",
                    "title": receipt.get("product_name", "未命名配方"),
                    "desc": receipt.get("product_concept", ""),
                    "keyword": "recipe",
                    "data": receipt,
                }
                if entry not in favs:
                    favs.append(entry)
                    _save_favorites(favs)
                    st.session_state.favorites = favs
                    st.toast(f"已收藏：{entry['title']}")
                else:
                    st.toast("此配方已在收藏清單中。")

        with col_dl:
            st.download_button(
                "⬇️ 下載 JSON",
                data=json.dumps(receipt, ensure_ascii=False, indent=2),
                file_name=f"{receipt.get('product_name','receipt')}.json",
                mime="application/json",
                use_container_width=True,
            )

        with col_reg:
            if st.button("🛡️ 送法規分析", use_container_width=True):
                st.session_state.concept_input = json.dumps(receipt, ensure_ascii=False, indent=2)
                st.success("已送到「深度研發與法規 AI」頁面，請切換後點擊分析。")
