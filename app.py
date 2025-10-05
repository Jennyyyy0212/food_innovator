import os
import json
import re
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
from google import genai  # ✅ 新版 SDK

# -----------------------------------------------------------
# Page setup
# -----------------------------------------------------------
st.set_page_config(page_title="AI 食品靈感引擎 (Gemini + Streamlit)", page_icon="🌿", layout="wide")
st.title("🌿 AI 食品靈感引擎（便利店研發 · 互動版）")
st.caption("Gemini 2.5 + Streamlit · 從關鍵字到靈感樹與研發八問分析")

# -----------------------------------------------------------
# API key
# -----------------------------------------------------------
api_key = "AIzaSyDOMU79t-5AOIIN0MIWbCTK6NiB3uq1PeM"
with st.sidebar:
    st.header("🔐 API 設定")
    sidebar_key = st.text_input("GOOGLE_API_KEY（不會儲存）", value=api_key, type="password")
    model_name = st.selectbox("Gemini 模型", ["gemini-2.5-flash"], index=0)
    st.markdown("---")
    st.caption("到 Google AI Studio 取得金鑰 → https://aistudio.google.com/")

if not sidebar_key:
    st.info("請在左側輸入 GOOGLE_API_KEY 以啟用 Gemini。")
    st.stop()

# ✅ 新 SDK 初始化
try:
    client = genai.Client(api_key=sidebar_key)
except Exception as e:
    st.error(f"❌ 初始化 Gemini 失敗：{e}")
    st.stop()

# -----------------------------------------------------------
# Sidebar test button
# -----------------------------------------------------------
with st.sidebar:
    st.markdown("---")
    st.subheader("🧪 測試連線")
    if st.button("🔍 測試 Gemini API 連線"):
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents="請用一句話描述亞洲便利店飲品創新趨勢。"
            )
            st.success("✅ 連線成功！Gemini 回覆：")

            # --- safer text extraction across SDK versions ---
            text_out = None
            if hasattr(resp, "text") and resp.text:
                st.write("[DEBUG]", "resp.text")
                text_out = resp.text
            elif getattr(resp, "candidates", None):
                st.write("[DEBUG]", "resp.candidates[0].content.parts[0].text")
                try:
                    text_out = resp.candidates[0].content.parts[0].text
                except Exception:
                    text_out = str(resp.candidates[0])
            elif hasattr(resp, "output_text"):
                st.write("[DEBUG]", "output_text")
                text_out = resp.output_text
            else:
                text_out = str(resp)

            st.write(text_out or "⚠️ （Gemini 回覆為空，可能是模型暫無輸出）")

        except Exception as e:
            st.error(f"❌ 連線測試失敗：{e}")

# -----------------------------------------------------------
# Helper functions
# -----------------------------------------------------------
def parse_json_loose(text: str) -> Any:
    """強化版 JSON 解析：自動清掉 Markdown 與亂碼"""
    import json, re
    if not text:
        return {"nodes": [{"title": "Empty", "desc": "", "children": []}]}

    # 清除 Markdown 符號與反引號
    cleaned = re.sub(r"```(?:json)?", "", text)
    cleaned = cleaned.replace("```", "").strip()

    # 嘗試直接解析
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 若有多餘說明文字，嘗試抽取中括號或大括號部分
    match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.S)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    return {"nodes": [{"title": "Raw Output", "desc": cleaned[:200], "children": []}]}



def ensure_node_shape(data: Any, keyword: str = "") -> List[Dict[str, Any]]:
    """通用解析器：支援多層 JSON 結構（root / levels / ideas / sub_ideas / children）
    可解析多種 Prompt 格式，並自動轉換為 title/desc/children 統一格式。
    """
    if not data:
        return []

    # --- case 1: dict (最外層物件) ---
    if isinstance(data, dict):
        # 若包含 root/nodes/levels/ideas/children
        if "root" in data:
            return ensure_node_shape(data["root"], keyword)
        if "nodes" in data:
            return ensure_node_shape(data["nodes"], keyword)
        if "levels" in data:
            return ensure_node_shape(data["levels"], keyword)
        if "ideas" in data:
            return ensure_node_shape(data["ideas"], keyword)
        if "children" in data and isinstance(data["children"], list):
            # 如果是一個符合 title/desc/children 結構的節點
            title = data.get("title") or data.get("name") or data.get("idea") or keyword or "未命名節點"
            desc = data.get("desc", "")
            return [{
                "title": title,
                "desc": desc,
                "children": ensure_node_shape(data["children"], keyword)
            }]

        # 若為一般 dict -> 每個 key 都視為節點
        nodes = []
        for k, v in data.items():
            nodes.append({
                "title": str(k),
                "desc": "" if isinstance(v, (dict, list)) else str(v),
                "children": ensure_node_shape(v, keyword) if isinstance(v, (dict, list)) else []
            })
        return nodes

    # --- case 2: list (root 或子層 list) ---
    if isinstance(data, list):
        nodes = []
        for item in data:
            # dict 節點
            if isinstance(item, dict):
                title = item.get("title") or item.get("name") or item.get("idea") or "未命名節點"
                desc = item.get("desc", "")
                children = None

                # 嘗試從各種鍵取 children
                for key in ["children", "ideas", "sub_ideas", "levels", "nodes"]:
                    if key in item and isinstance(item[key], list):
                        children = ensure_node_shape(item[key], keyword)
                        break

                nodes.append({
                    "title": title,
                    "desc": desc,
                    "children": children if children is not None else []
                })
            # 純文字節點
            elif isinstance(item, str):
                nodes.append({"title": item, "desc": "", "children": []})
            # 其他類型 (fallback)
            else:
                nodes.append({"title": str(item), "desc": "", "children": []})
        return nodes

    # --- case 3: 純文字或其他型態 ---
    return [{"title": str(data), "desc": "", "children": []}]



def gemini_generate(prompt: str) -> str:
    """使用新版 SDK 生成內容"""
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        # 新版 SDK 的 response.text 可能在 response.candidates[0].content.parts[0].text 中
        text_out = None
        if hasattr(response, "text") and response.text:
            text_out = response.text
        elif getattr(response, "candidates", None):
            try:
                text_out = response.candidates[0].content.parts[0].text
            except Exception:
                text_out = str(response.candidates[0])
        elif hasattr(response, "output_text"):
            text_out = response.output_text
        else:
            text_out = str(response)

        # --- clean Markdown fences ---
        cleaned = (
            text_out
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        # --- debug preview ---
        st.write("🪄 Gemini 原始輸出 (清理後)：")
        st.code(cleaned[:500] + ("..." if len(cleaned) > 500 else ""), language="json")

        return cleaned
    except Exception as e:
        st.error(f"❌ Gemini 錯誤：{e}")
        return ""

# -----------------------------------------------------------
# Prompts
# -----------------------------------------------------------
BASE_PROMPT = """
你是一位亞洲便利店食品研發顧問兼趨勢設計師。
請根據關鍵字「{keyword}」，產生一棵「食品研發靈感樹」的 JSON，包含以下層級

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
請根據整體產品主題「{keyword}」，並結合當前節點「{title}: {desc}」，
延伸出 3～6 個相關子靈感（需與主題與上層方向保持一致）。

請以純 JSON 陣列格式輸出，格式如下：
[
  {{"title": "子靈感A", "desc": "簡要說明", "children": []}},
  {{"title": "子靈感B", "desc": "簡要說明", "children": []}}
]

要求：
- 子靈感應與「{keyword}」主題直接相關。
- 保持便利店食品研發語境（例如口味、形態、包裝、食用場景、族群趨勢等）。
- 不要包含說明文字或 Markdown，只輸出 JSON。
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
# Session state
# -----------------------------------------------------------
if "idea_tree" not in st.session_state:
    st.session_state.idea_tree = []
if "keyword" not in st.session_state:
    st.session_state.keyword = ""
if "report_md" not in st.session_state:
    st.session_state.report_md = ""
if "rd_analysis" not in st.session_state:
    st.session_state.rd_analysis = ""

# -----------------------------------------------------------
# Sidebar inputs
# -----------------------------------------------------------
with st.sidebar:
    st.header("🎛️ 互動控制")
    keyword = st.text_input("🔑 輸入關鍵字（例：抹茶 / 紫薯 / 氣泡）", value=st.session_state.keyword)
    col1, col2 = st.columns(2)
    with col1:
        gen_btn = st.button("🌟 生成靈感樹", use_container_width=True)
    with col2:
        clr_btn = st.button("🧹 清空", use_container_width=True)

if clr_btn:
    st.session_state.idea_tree = []
    st.session_state.keyword = ""
    st.session_state.report_md = ""
    st.session_state.rd_analysis = ""
    st.rerun()

# -----------------------------------------------------------
# Generate base tree
# -----------------------------------------------------------
if gen_btn and keyword.strip():
    st.write("✅ [DEBUG] 生成靈感樹按鈕被按下")
    st.session_state.keyword = keyword.strip()
    with st.spinner("Gemini 正在生成靈感樹..."):
        text = gemini_generate(BASE_PROMPT.format(keyword=keyword))
    data = parse_json_loose(text)
    st.json(data)
    st.session_state.idea_tree = ensure_node_shape(data, keyword=st.session_state.keyword)

# -----------------------------------------------------------
# Node rendering
# -----------------------------------------------------------
def remove_by_path(idx_path: str):
    parts = [int(p) for p in idx_path.split(".")]
    def rec(nodes, path):
        if len(path) == 1:
            del nodes[path[0]]
            return
        rec(nodes[path[0]]["children"], path[1:])
    rec(st.session_state.idea_tree, parts)

def render_node(node: Dict[str, Any], level=0, idx_path="0"):
    pad = 8 * level
    with st.container():
        st.markdown(f"<div style='margin-left:{pad}px'></div>", unsafe_allow_html=True)
        with st.expander(f"▾ **{node['title']}**", expanded=(level == 0)):
            st.write(node.get("desc", ""))
            cols = st.columns([1,1,1,4])
            with cols[0]:
                if st.button("➕ 深入", key=f"expand_{idx_path}"):
                    with st.spinner("延伸子靈感..."):
                        text = gemini_generate(EXPAND_PROMPT.format(
                            keyword=st.session_state.keyword,
                            title=node['title'],
                            desc=node.get('desc', '')
                        ))
                    data = parse_json_loose(text)
                    node.setdefault("children", []).extend(ensure_node_shape(data))
            with cols[1]:
                if st.button("⭐ 收藏", key=f"fav_{idx_path}"):
                    st.toast(f"已收藏：{node['title']}")
            with cols[2]:
                if st.button("🗑️ 移除", key=f"rm_{idx_path}"):
                    remove_by_path(idx_path)
                    st.rerun()

            for j, child in enumerate(node.get("children", [])):
                render_node(child, level + 1, f"{idx_path}.{j}")

# -----------------------------------------------------------
# Main render
# -----------------------------------------------------------
if st.session_state.idea_tree:
    st.subheader(f"🌳 靈感樹：{st.session_state.keyword}")
    for i, n in enumerate(st.session_state.idea_tree):
        render_node(n, 0, str(i))

    st.markdown("---")
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        st.download_button(
            "⬇️ 下載 JSON",
            data=json.dumps({"keyword": st.session_state.keyword, "nodes": st.session_state.idea_tree}, ensure_ascii=False, indent=2),
            file_name=f"idea_tree_{st.session_state.keyword}.json",
            mime="application/json"
        )
    with c2:
        if st.button("📋 生成研发八问分析"):
            with st.spinner("Gemini 正在分析研發八問..."):
                rd_text = gemini_generate(RD_PROMPT.format(
                    json_payload=json.dumps({"keyword": st.session_state.keyword, "nodes": st.session_state.idea_tree}, ensure_ascii=False)
                ))
            st.session_state.rd_analysis = rd_text
    with c3:
        if st.button("📝 匯出 Markdown 報告"):
            with st.spinner("Gemini 正在整理報告..."):
                md = gemini_generate(REPORT_PROMPT.format(
                    json_payload=json.dumps({"keyword": st.session_state.keyword, "nodes": st.session_state.idea_tree}, ensure_ascii=False)
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
