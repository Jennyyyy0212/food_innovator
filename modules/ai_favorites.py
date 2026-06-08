import json
import streamlit as st

from gemini_utils import gemini_generate as _gemini_generate
from modules.ai_innovation import _load_favorites, _save_favorites, REPORT_PROMPT
from modules.ai_receipt import recipe_to_markdown


def render_favorites(client, model_name):
    st.title("⭐ 收藏清單")
    st.caption("靈感節點與食品配方收藏，可下載或送到其他功能頁面。")

    if not st.session_state.get("_fav_ls_loaded"):
        loaded = _load_favorites()
        if loaded is not None:
            st.session_state.favorites = loaded
            st.session_state._fav_ls_loaded = True
    if "favorites" not in st.session_state:
        st.session_state.favorites = []

    favs = st.session_state.favorites

    # -----------------------------------------------------------
    # Top action bar
    # -----------------------------------------------------------
    col_dl, col_clear, col_spacer = st.columns([1, 1, 4])
    with col_dl:
        st.download_button(
            "⬇️ 下載全部收藏 JSON",
            data=json.dumps(favs, ensure_ascii=False, indent=2),
            file_name="favorites.json",
            mime="application/json",
            disabled=not favs,
            use_container_width=True,
        )
    with col_clear:
        if st.button("🗑️ 清空收藏", disabled=not favs, use_container_width=True):
            st.session_state.favorites = []
            _save_favorites([])
            st.rerun()

    st.divider()

    if not favs:
        st.info(
            "尚無收藏。\n\n"
            "- 在「💡 食品靈感生成 AI」點擊節點上的 ⭐ 收藏 儲存靈感\n"
            "- 在「🧾 食譜生成 AI」點擊 ⭐ 加入收藏 儲存配方"
        )
        return

    # Group by type for count display
    idea_count   = sum(1 for f in favs if f.get("type") != "recipe")
    recipe_count = sum(1 for f in favs if f.get("type") == "recipe")
    parts = []
    if idea_count:
        parts.append(f"💡 靈感節點 {idea_count} 項")
    if recipe_count:
        parts.append(f"🧾 配方 {recipe_count} 項")
    st.caption("共 " + "、".join(parts))

    # -----------------------------------------------------------
    # Cards
    # -----------------------------------------------------------
    for i, fav in enumerate(favs):
        fav_type = fav.get("type", "idea")

        with st.container(border=True):
            if fav_type == "recipe":
                _render_recipe_card(fav, i)
            else:
                _render_idea_card(fav, i, client, model_name)


# -----------------------------------------------------------
# Card renderers
# -----------------------------------------------------------

def _render_subtree(node: dict, keyword: str, card_idx: int, level: int = 0, path: str = ""):
    """Recursively render child nodes as nested expanders with action buttons."""
    for j, child in enumerate(node.get("children", [])):
        node_path = f"{path}.{j}" if path else str(j)
        indent_px = 16 * level
        st.markdown(f"<div style='margin-left:{indent_px}px'>", unsafe_allow_html=True)
        with st.expander(f"▾ **{child['title']}**", expanded=False):
            if child.get("desc"):
                st.caption(child["desc"])

            btn_fav, btn_recipe = st.columns(2)
            with btn_fav:
                if st.button("⭐ 收藏此節點", key=f"sub_fav_{card_idx}_{node_path}", use_container_width=True):
                    entry = {"type": "idea", "title": child["title"], "desc": child.get("desc", ""), "keyword": keyword, "data": child}
                    favs = st.session_state.favorites
                    existing = {(f.get("type", "idea"), f.get("title", ""), f.get("keyword", "")) for f in favs}
                    if ("idea", child["title"], keyword) not in existing:
                        favs.append(entry)
                        _save_favorites(favs)
                        st.toast(f"已收藏：{child['title']}")
                    else:
                        st.toast(f"已在收藏清單中：{child['title']}")
            with btn_recipe:
                if st.button("🧾 送配方", key=f"sub_recipe_{card_idx}_{node_path}", use_container_width=True):
                    st.session_state.receipt_concept_prefill = (
                        f"{child['title']}：{child.get('desc', '')}" if child.get("desc") else child["title"]
                    )
                    st.toast(f"已送出「{child['title']}」→ 請切到「🧾 食譜生成 AI」頁面。")
                    st.rerun()

            if child.get("children"):
                _render_subtree(child, keyword, card_idx, level + 1, node_path)
        st.markdown("</div>", unsafe_allow_html=True)


def _render_idea_card(fav: dict, i: int, client, model_name):
    header_col, btn_col = st.columns([5, 1])
    with header_col:
        st.markdown(f"### 💡 {fav['title']}")
        if fav.get("keyword"):
            st.caption(f"關鍵字：`{fav['keyword']}`")
    with btn_col:
        if st.button("✕", key=f"fav_rm_{i}", use_container_width=True, help="移除"):
            st.session_state.favorites.pop(i)
            _save_favorites(st.session_state.favorites)
            st.rerun()

    if fav.get("desc"):
        st.write(fav["desc"])

    node_data = fav.get("data") or {"title": fav["title"], "desc": fav.get("desc", ""), "keyword": fav.get("keyword", "")}
    children = node_data.get("children", [])
    if children:
        with st.expander(f"📂 子節點（{len(children)} 項）", expanded=False):
            _render_subtree(node_data, keyword=fav.get("keyword", ""), card_idx=i)

    # Action buttons
    act_dl, act_report, act_recipe = st.columns(3)

    with act_dl:
        st.download_button(
            "⬇️ 下載 JSON",
            data=json.dumps(node_data, ensure_ascii=False, indent=2),
            file_name=f"{fav['title']}.json",
            mime="application/json",
            key=f"fav_idea_dl_{i}",
            use_container_width=True,
        )

    with act_report:
        if st.button("📄 生成研發報告", key=f"fav_idea_report_{i}", use_container_width=True):
            prompt = REPORT_PROMPT.format(
                keyword=fav.get("keyword") or fav["title"],
                json_payload=json.dumps(node_data, ensure_ascii=False, indent=2),
            )
            with st.spinner("AI 正在生成研發報告…"):
                md = _gemini_generate(
                    client, model_name, prompt,
                    on_retry=lambda attempt, delay: st.toast(f"⏳ 重試第 {attempt + 1} 次…"),
                    on_error=lambda e: st.error(f"❌ Gemini 錯誤：{e}"),
                )
            st.session_state[f"idea_report_{i}"] = md

    with act_recipe:
        if st.button("🧾 送到食譜生成器", key=f"fav_to_recipe_{i}", use_container_width=True):
            st.session_state.receipt_concept_prefill = (
                f"{fav['title']}：{fav.get('desc', '')}" if fav.get("desc") else fav["title"]
            )
            st.toast(f"已送出「{fav['title']}」→ 請切到「🧾 食譜生成 AI」頁面。")
            st.rerun()

    # Generated report display
    report_md = st.session_state.get(f"idea_report_{i}", "")
    if report_md:
        st.divider()
        with st.expander("📄 研發報告", expanded=True):
            st.markdown(report_md)
            st.download_button(
                "⬇️ 下載報告 .md",
                data=report_md,
                file_name=f"{fav['title']}_研發報告.md",
                mime="text/markdown",
                key=f"fav_idea_md_dl_{i}",
                use_container_width=True,
            )


def _render_recipe_card(fav: dict, i: int):
    receipt = fav.get("data", {})

    header_col, btn_col = st.columns([5, 1])
    with header_col:
        st.markdown(f"### 🧾 {fav['title']}")
        if fav.get("desc"):
            st.caption(fav["desc"])
    with btn_col:
        if st.button("✕", key=f"fav_rm_{i}", use_container_width=True, help="移除"):
            st.session_state.favorites.pop(i)
            _save_favorites(st.session_state.favorites)
            st.rerun()

    # Compact ingredient summary
    ingredients = receipt.get("ingredients", [])
    additives   = receipt.get("additives", [])
    if ingredients:
        names = "、".join(ing.get("name", "") for ing in ingredients[:6])
        suffix = f"…等 {len(ingredients)} 項" if len(ingredients) > 6 else ""
        st.caption(f"主要食材：{names}{suffix}")
    if additives:
        add_names = "、".join(a.get("name", "") for a in additives)
        st.caption(f"添加物：{add_names}")

    # Full markdown view in expander
    with st.expander("📋 查看完整配方", expanded=False):
        st.markdown(recipe_to_markdown(receipt))

    # Action row
    act_dl, act_reg, act_concept = st.columns(3)
    with act_dl:
        st.download_button(
            "⬇️ 下載 JSON",
            data=json.dumps(receipt, ensure_ascii=False, indent=2),
            file_name=f"{fav['title']}.json",
            mime="application/json",
            key=f"fav_dl_{i}",
            use_container_width=True,
        )
    with act_reg:
        if st.button("🛡️ 送法規分析", key=f"fav_reg_{i}", use_container_width=True):
            st.session_state.concept_input = json.dumps(receipt, ensure_ascii=False, indent=2)
            st.toast("已送到「🔬 深度研發與法規 AI」頁面。")
            st.rerun()
    with act_concept:
        if st.button("💡 送靈感生成", key=f"fav_idea_{i}", use_container_width=True,
                     help="以產品名稱作為靈感樹關鍵字"):
            st.session_state.keyword = fav["title"]
            st.toast(f"已送出「{fav['title']}」→ 請切到「💡 食品靈感生成 AI」頁面。")
            st.rerun()
