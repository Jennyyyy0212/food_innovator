import json
import streamlit as st
from typing import Any, Dict, List

from gemini_utils import gemini_generate as _gemini_generate, parse_json_loose
from modules.ai_chat import render_chat_panel

_LS_KEY = "food_innovator_favorites"


def _load_favorites():
    """Return list of favorites from browser localStorage, or None if not ready yet."""
    ls = st.session_state.get("_ls")
    if ls is None:
        return None
    raw = ls.getItem(_LS_KEY)
    if raw is None:
        return None  # localStorage hasn't responded yet (first render)
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_favorites(favs: List[Dict]) -> None:
    ls = st.session_state.get("_ls")
    if ls is not None:
        ls.setItem(_LS_KEY, json.dumps(favs, ensure_ascii=False))


def _node_to_fav(node: Dict, keyword: str) -> Dict:
    """Build a single favorite entry for a node, embedding its full subtree in 'data'."""
    return {"type": "idea", "title": node["title"], "desc": node.get("desc", ""), "keyword": keyword, "data": node}

# -----------------------------------------------------------
# Module-level prompt templates
# -----------------------------------------------------------

BASE_PROMPT = """
你是一位亞洲便利店食品研發顧問兼趨勢設計師。
請根據以下輸入主題或句子「{keyword}」，產生一棵「食品研發靈感樹」的 JSON，包含以下層級

請嚴格遵循以下結構格式（所有層級的鍵名都要一致）：
{{
"root": [
    {{
    "title": "1) 主題探索 (Theme Exploration)",
    "desc": "這一層描述主題方向，例如市場趨勢與概念開發。",
    "children": [
        {{
        "title": "經典咖哩再現",
        "desc": "融合亞洲風味的懷舊創新主題。",
        "children": [
            {{"title": "日式豬排咖哩飯", "desc": "濃厚甜味、適合上班族午餐。", "children": []}},
            {{"title": "泰式綠咖哩雞", "desc": "微辣清爽、主打異國口味。", "children": []}},
            {{"title": "南洋叻沙咖哩麵", "desc": "湯麵型態、熱銷於便利商店。", "children": []}}
        ]
        }}
    ]
    }},
    {{
    "title": "2) 食材靈感 (Ingredient Inspiration)",
    "desc": "探索原料、風味與組合的可能性。",
    "children": [
        {{"title": "植物肉咖哩", "desc": "迎合健康潮流與永續概念。", "children": []}},
        {{"title": "海鮮椰香咖哩", "desc": "以椰奶中和辛香，帶出南洋風味。", "children": []}}
    ]
    }},
    {{
    "title": "3) 形狀設計 (Shape Design)",
    "desc": "考慮產品形態、便攜性與創意造型。",
    "children": []
    }},
    {{
    "title": "4) 包裝創意 (Packaging Creativity)",
    "desc": "設計與便利性兼具的包裝靈感。",
    "children": []
    }},
    {{
    "title": "5) 食用方式 (Eating Method)",
    "desc": "探索不同的食用場景與體驗方式。",
    "children": []
    }},
    {{
    "title": "6) 大眾化分析 (Popularization Analysis)",
    "desc": "分析市場接受度、消費族群與趨勢。",
    "children": []
    }}
]
}}

注意：
- 所有節點都只能使用 `"title"`, `"desc"`, `"children"` 三個鍵。
- 不要出現 `"idea"`, `"sub_ideas"`, `"levels"`, 或 `"root"` 等鍵。
- 必須輸出有效 JSON（以 `{{` 開頭、以 `}}` 結尾），不得包含說明文字或 Markdown 標籤。
"""

EXPAND_PROMPT = """
你是一位亞洲便利店食品創新顧問。
請根據以下脈絡，延伸出具體的 3～5 個新靈感，並以 JSON 陣列格式輸出。

【產品主題】{keyword}
【目前節點】{title}: {desc}
【脈絡層級】{context}
【深入方向】{deep_dive}

要求：
- 子靈感必須同時與「{keyword}」、「{title}」、「{context}」和及「{deep_dive}」方向相關（若「{deep_dive}」為空，則僅依主題延伸）。
2. 每個子靈感應該延伸該節點的核心意涵，可包括：
- 更具體的應用情境、方法、實踐或案例；
- 若屬抽象主題，可展開在理論、策略或框架層面；
- 若屬實體主題，可展開為具體方案、服務、體驗或產品。
3. 請結合便利店環境考量（保存、展示、組合包裝、客群互動）。
4. 每個子靈感包含：
- title：產品名稱或概念
- desc：具體描述，包括組成、包裝方式、使用情境與風味體驗
- children：空陣列 []

請以純 JSON 陣列格式輸出，格式如下：
[
{{"title": "子靈感A", "desc": "簡要說明", "children": []}},
{{"title": "子靈感B", "desc": "簡要說明", "children": []}}
]
"""

REPORT_PROMPT = """
你是一位食品研發企劃。
請將以下「靈感樹 JSON」整理成 Markdown 研發報告。
報告結構：
# 產品研發報告：{keyword}
## 一、主題概述
## 二、靈感層級摘要
## 三、市場與法規洞察
## 四、後續研發方向

JSON：
{json_payload}

請務必輸出有效 Markdown，不要輸出 JSON 或多餘格式說明。
"""

RD_PROMPT = """
你是一位食品研發專家。
請根據以下靈感樹內容（JSON），回答這八個問題，以條列格式輸出：
1. 產品定位是什麼？適合哪個消費族群？
2. 有無競品？與現有市場產品差異在哪？
3. 成本控制重點在哪？哪些原料可能影響成本？
4. 現有工藝是否能實現？有何生產挑戰？
5. 核心原料是什麼？來源與穩定性如何？
6. 推薦的包裝形式是什麼？設計建議？
7. 企劃面有什麼潛在主題或市場訴求？
8. 需參考哪些食品標準或法規？

JSON：
{json_payload}

請務必輸出純文字或 Markdown 條列清單，勿包含多餘符號。
"""

# -----------------------------------------------------------
# Module-level helpers
# -----------------------------------------------------------

def ensure_node_shape(data: Any, keyword: str = "") -> List[Dict[str, Any]]:
    if not data:
        return []
    if isinstance(data, dict):
        for key in ("root", "nodes", "levels", "ideas"):
            if key in data:
                return ensure_node_shape(data[key], keyword)
        if "children" in data and isinstance(data["children"], list):
            title = data.get("title") or data.get("name") or data.get("idea") or keyword or "未命名節點"
            return [{"title": title, "desc": data.get("desc", ""), "children": ensure_node_shape(data["children"], keyword)}]
        return [
            {
                "title": str(k),
                "desc": "" if isinstance(v, (dict, list)) else str(v),
                "children": ensure_node_shape(v, keyword) if isinstance(v, (dict, list)) else [],
            }
            for k, v in data.items()
        ]
    if isinstance(data, list):
        nodes = []
        for item in data:
            if isinstance(item, dict):
                title = item.get("title") or item.get("name") or item.get("idea") or "未命名節點"
                children = None
                for key in ("children", "ideas", "sub_ideas", "levels", "nodes"):
                    if key in item and isinstance(item[key], list):
                        children = ensure_node_shape(item[key], keyword)
                        break
                nodes.append({"title": title, "desc": item.get("desc", ""), "children": children or []})
            elif isinstance(item, str):
                nodes.append({"title": item, "desc": "", "children": []})
            else:
                nodes.append({"title": str(item), "desc": "", "children": []})
        return nodes
    return [{"title": str(data), "desc": "", "children": []}]


# -----------------------------------------------------------
# Main render
# -----------------------------------------------------------

def render_innovation(client, model_name):
    st.title("🌿 AI 食品靈感引擎（便利店研發 · 互動版）")
    st.caption("Gemini 2.5 + Streamlit · 從關鍵字到靈感樹與研發八問分析")

    # Session state init
    for key, default in [
        ("open_path", None),
        ("idea_tree", []),
        ("keyword", ""),
        ("report_md", ""),
        ("rd_analysis", ""),
        ("chat_open", False),
        ("chat_messages", []),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default
    # Keep retrying until localStorage responds (returns None on the first render).
    if not st.session_state.get("_fav_ls_loaded"):
        loaded = _load_favorites()
        if loaded is not None:
            st.session_state.favorites = loaded
            st.session_state._fav_ls_loaded = True
    if "favorites" not in st.session_state:
        st.session_state.favorites = []

    def gemini_generate(prompt: str) -> str:
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
    # Sidebar inputs
    # -----------------------------------------------------------
    with st.sidebar:
        st.header("🎛️ 互動控制")
        keyword = st.text_input("🔑 輸入關鍵字（例：抹茶 / 紫薯 / 氣泡）", value=st.session_state.keyword)
        col1, col2 = st.columns(2)
        with col1:
            gen_btn = st.button("🌟 生成靈感樹", width="stretch")
        with col2:
            clr_btn = st.button("🧹 清空", width="stretch")

    if clr_btn:
        for key in ("idea_tree", "keyword", "report_md", "rd_analysis", "open_path"):
            st.session_state[key] = [] if key == "idea_tree" else None if key == "open_path" else ""
        st.rerun()

    # -----------------------------------------------------------
    # Generate base tree
    # -----------------------------------------------------------
    if gen_btn and keyword.strip():
        st.session_state.keyword = keyword.strip()
        with st.spinner("Gemini 正在生成靈感樹..."):
            text = gemini_generate(BASE_PROMPT.format(keyword=keyword))
        data = parse_json_loose(text)
        st.session_state.idea_tree = ensure_node_shape(data, keyword=st.session_state.keyword)

    # -----------------------------------------------------------
    # Node renderer
    # -----------------------------------------------------------
    def remove_by_path(idx_path: str):
        parts = [int(p) for p in idx_path.split(".")]
        def rec(nodes, path):
            if len(path) == 1:
                del nodes[path[0]]
                return
            rec(nodes[path[0]]["children"], path[1:])
        rec(st.session_state.idea_tree, parts)

    def render_node(node: Dict[str, Any], level=0, idx_path="0", parent_titles=None):
        parent_titles = parent_titles or []

        open_path = st.session_state.open_path

        def _should_be_open() -> bool:
            if level == 0:
                return True
            if not open_path:
                return False
            op = open_path.split(".")
            ip = idx_path.split(".")
            # True when this node is an ancestor of, equal to, or a descendant of open_path
            return op[:len(ip)] == ip or ip[:len(op)] == op

        # on_click fires BEFORE the script reruns, so the expander sees the
        # updated open_path on the very same render that follows the click.
        def _mark_open():
            st.session_state.open_path = idx_path

        with st.container():
            st.markdown(f"<div style='margin-left:{20 * level}px'></div>", unsafe_allow_html=True)
            with st.expander(f"▾ **{node['title']}**", expanded=_should_be_open()):
                st.write(node.get("desc", "無描述"))
                deep_col = st.text_input(
                    "💡 想深入探討什麼？（可留空使用預設）",
                    key=f"deep_input_{idx_path}",
                    placeholder="例如：永續包裝 / 新口味創新",
                )
                cols = st.columns([1, 1, 1, 1])
                with cols[0]:
                    if st.button("➕ 深入", key=f"expand_{idx_path}", on_click=_mark_open):
                        with st.spinner("延伸子靈感..."):
                            text = gemini_generate(EXPAND_PROMPT.format(
                                keyword=st.session_state.keyword,
                                deep_dive=deep_col.strip() or "（無特定方向）",
                                title=node["title"],
                                desc=node.get("desc", ""),
                                context=" > ".join(parent_titles + [node["title"]]),
                            ))
                        node.setdefault("children", []).extend(ensure_node_shape(parse_json_loose(text)))
                        st.rerun()
                with cols[1]:
                    if st.button("⭐ 收藏", key=f"fav_{idx_path}"):
                        entry = _node_to_fav(node, st.session_state.keyword)
                        favs = st.session_state.favorites
                        existing = {(f.get("type", "idea"), f.get("title", ""), f.get("keyword", "")) for f in favs}
                        key = ("idea", entry["title"], entry["keyword"])
                        if key not in existing:
                            favs.append(entry)
                            _save_favorites(favs)
                            child_count = len(node.get("children", []))
                            suffix = f"（含 {child_count} 個子節點）" if child_count else ""
                            st.toast(f"已收藏：{node['title']}{suffix}")
                        else:
                            st.toast(f"已在收藏清單中：{node['title']}")
                with cols[2]:
                    if st.button("🧾 送配方", key=f"to_recipe_{idx_path}"):
                        st.session_state.receipt_concept_prefill = (
                            f"{node['title']}：{node.get('desc', '')}" if node.get("desc") else node["title"]
                        )
                        st.toast(f"已送出「{node['title']}」→ 請切到「🧾 食譜生成 AI」頁面。")
                with cols[3]:
                    if st.button("🗑️ 移除", key=f"rm_{idx_path}"):
                        remove_by_path(idx_path)
                        st.rerun()
                for j, child in enumerate(node.get("children", [])):
                    render_node(child, level + 1, f"{idx_path}.{j}", parent_titles + [node["title"]])

    # -----------------------------------------------------------
    # Floating chat toggle button (bottom-right, pure CSS)
    #
    # The sentinel <span> placed immediately before the button lets
    # us target that exact button container with an adjacent-sibling
    # CSS selector and apply position:fixed — no extra package needed.
    # -----------------------------------------------------------
    st.markdown(
        """
        <style>
        /* Sentinel → button container: make it float */
        div[data-testid="stMarkdownContainer"]:has(.chat-fab-sentinel)
            + div[data-testid="stButton"] {
            position: fixed !important;
            bottom: 3rem;
            right: 1.5rem;
            z-index: 9999;
            width: auto !important;
        }
        /* Style the button itself as a circle */
        div[data-testid="stMarkdownContainer"]:has(.chat-fab-sentinel)
            + div[data-testid="stButton"] button {
            border-radius: 50% !important;
            width: 52px !important;
            height: 52px !important;
            min-height: unset !important;
            padding: 0 !important;
            font-size: 22px !important;
            line-height: 1 !important;
            box-shadow: 0 4px 14px rgba(0,0,0,0.28) !important;
        }
        </style>
        <span class="chat-fab-sentinel"></span>
        """,
        unsafe_allow_html=True,
    )
    chat_is_open = st.session_state.chat_open
    if st.button(
        "✕" if chat_is_open else "💬",
        key="chat_fab",
        type="primary",
        help="關閉 AI 顧問" if chat_is_open else "開啟 AI 顧問",
    ):
        st.session_state.chat_open = not chat_is_open
        st.rerun()

    # -----------------------------------------------------------
    # Main render — split layout when chat is open
    # -----------------------------------------------------------
    if st.session_state.chat_open:
        tree_col, chat_col = st.columns([0.60, 0.40], gap="medium")
    else:
        tree_col = st.container()
        chat_col = None

    with tree_col:
        if st.session_state.idea_tree:
            st.subheader(f"🌳 靈感樹：{st.session_state.keyword}")
            for i, n in enumerate(st.session_state.idea_tree):
                render_node(n, 0, str(i))

            st.markdown("---")
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                st.download_button(
                    "⬇️ 下載 JSON",
                    data=json.dumps(
                        {"keyword": st.session_state.keyword, "nodes": st.session_state.idea_tree},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    file_name=f"idea_tree_{st.session_state.keyword}.json",
                    mime="application/json",
                )
            with c2:
                if st.button("📋 生成研发八问分析"):
                    with st.spinner("Gemini 正在分析研發八問..."):
                        rd_text = gemini_generate(RD_PROMPT.format(
                            json_payload=json.dumps(
                                {"keyword": st.session_state.keyword, "nodes": st.session_state.idea_tree},
                                ensure_ascii=False,
                            )
                        ))
                    st.session_state.rd_analysis = rd_text
            with c3:
                if st.button("📝 匯出 Markdown 報告"):
                    with st.spinner("Gemini 正在整理報告..."):
                        md = gemini_generate(REPORT_PROMPT.format(
                            keyword=st.session_state.keyword,
                            json_payload=json.dumps(
                                {"keyword": st.session_state.keyword, "nodes": st.session_state.idea_tree},
                                ensure_ascii=False,
                            ),
                        ))
                    st.session_state.report_md = md

            if st.session_state.rd_analysis:
                st.subheader("🧪 研發八問分析")
                st.markdown(st.session_state.rd_analysis)

            if st.session_state.report_md:
                st.subheader("📄 研發報告（Markdown）")
                st.markdown(st.session_state.report_md)
        else:
            st.info("輸入關鍵字後點『生成靈感樹』開始探索。")

    if chat_col:
        with chat_col:
            render_chat_panel(
                client,
                model_name,
                keyword=st.session_state.keyword,
                open_path=st.session_state.open_path,
                idea_tree=st.session_state.idea_tree,
            )
