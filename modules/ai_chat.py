"""Floating AI consultant chat panel for the innovation tab."""

import streamlit as st
from typing import Any, Dict, List, Optional

from gemini_utils import gemini_generate as _gemini_generate


_SYSTEM_PROMPT = """你是一位資深亞洲便利店食品研發顧問，專長包括：
- 食品原料、配方與製程工藝
- 台灣、日本、中國、東南亞各地食品法規與添加物標準
- 市場趨勢、消費者洞察、競品分析
- 成本控制與量產可行性

請根據對話脈絡與目前節點資訊，提供簡潔、專業的繁體中文回覆。
若問題超出食品研發範疇，請說明並引導回主題。"""


def get_node_by_path(tree: List[Dict], path_str: Optional[str]) -> Optional[Dict]:
    """Return the node at idx_path from the idea tree, or None."""
    if not path_str or not tree:
        return None
    try:
        parts = [int(p) for p in path_str.split(".")]
        node = tree[parts[0]]
        for p in parts[1:]:
            node = node["children"][p]
        return node
    except (IndexError, KeyError, TypeError):
        return None


def _build_prompt(messages: List[Dict], context_block: str) -> str:
    history = "\n".join(
        f"{'使用者' if m['role'] == 'user' else 'AI'}：{m['content']}"
        for m in messages
    )
    return f"{_SYSTEM_PROMPT}\n\n{context_block}\n\n對話記錄：\n{history}"


def render_chat_panel(
    client,
    model_name: str,
    keyword: str,
    open_path: Optional[str],
    idea_tree: List[Dict],
) -> None:
    """Render the AI consultant chat panel (right-side column)."""

    # -- Header --
    st.markdown("### 💬 AI 研發顧問")

    ctx_node = get_node_by_path(idea_tree, open_path)
    if ctx_node:
        st.caption(f"📌 目前節點：**{ctx_node['title']}**")
        if ctx_node.get("desc"):
            with st.expander("節點描述", expanded=False):
                st.write(ctx_node["desc"])
    elif keyword:
        st.caption(f"📌 主題：**{keyword}**")
    else:
        st.caption("請先在左側生成靈感樹，點擊節點後再提問。")

    col_clear, col_space = st.columns([1, 3])
    with col_clear:
        if st.button("🗑 清除紀錄", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()

    st.divider()

    # -- Chat history --
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    messages_container = st.container(height=420)
    with messages_container:
        if not st.session_state.chat_messages:
            st.info(
                "你可以問我任何食品研發問題，例如：\n\n"
                "- 這個成分在台灣法規上有限量嗎？\n"
                "- 這個概念大概的量產成本？\n"
                "- 有沒有類似的競品？"
            )
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # -- Input --
    user_input = st.chat_input("輸入問題…", key="ai_chat_input")
    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})

        ctx_parts = []
        if keyword:
            ctx_parts.append(f"主題關鍵字：{keyword}")
        if ctx_node:
            ctx_parts.append(f"目前討論節點：{ctx_node['title']}")
            if ctx_node.get("desc"):
                ctx_parts.append(f"節點描述：{ctx_node['desc']}")
        context_block = "\n".join(ctx_parts)

        prompt = _build_prompt(st.session_state.chat_messages, context_block)

        with messages_container:
            with st.chat_message("assistant"):
                with st.spinner("思考中…"):
                    reply = _gemini_generate(
                        client,
                        model_name,
                        prompt,
                        on_retry=lambda attempt, delay: st.toast(
                            f"⏳ 重試中（第 {attempt + 1} 次）…"
                        ),
                        on_error=lambda e: st.error(f"❌ Gemini 錯誤：{e}"),
                    )
                st.markdown(reply)

        st.session_state.chat_messages.append({"role": "assistant", "content": reply})
        st.rerun()
