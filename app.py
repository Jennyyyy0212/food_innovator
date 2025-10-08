import os
import json
import re
from typing import List, Dict, Any
import streamlit_nested_layout
import streamlit as st
import pandas as pd
from google import genai  # ✅ 新版 SDK

tab1, tab2 = st.tabs(["💡 食品靈感生成 AI", "🔬 深度研發與法規 AI"])

# -----------------------------------------------------------
# Page setup
# -----------------------------------------------------------
st.set_page_config(page_title="AI 食品靈感引擎 (Gemini + Streamlit)", page_icon="🌿", layout="wide")
with tab1:
    st.title("🌿 AI 食品靈感引擎（便利店研發 · 互動版）")
    st.caption("Gemini 2.5 + Streamlit · 從關鍵字到靈感樹與研發八問分析")

    # -----------------------------------------------------------
    # API key
    # -----------------------------------------------------------
    api_key = os.getenv("GOOGLE_API_KEY", "")
    with st.sidebar:
        st.header("🔐 API 設定")
        # Only show status — never the key itself
        if api_key != "":
            st.success("✅ 已設定 Gemini API 金鑰")
        else:
            st.warning("⚠️ 尚未設定 API 金鑰")


        model_name = st.selectbox("Gemini 模型", ["gemini-2.5-flash"], index=0)
        st.markdown("---")

        

        st.caption("到 Google AI Studio 取得金鑰 → https://aistudio.google.com/")


    if api_key == "":
        st.stop()  # stop the app until API key entered

    # ✅ 新 SDK 初始化
    try:
        client = genai.Client(api_key=api_key)
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
    if "open_path" not in st.session_state:
        st.session_state.open_path = None

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
            # st.write("🪄 Gemini 原始輸出 (清理後)：")
            # st.code(cleaned[:500] + ("..." if len(cleaned) > 500 else ""), language="json")

            return cleaned
        except Exception as e:
            st.error(f"❌ Gemini 錯誤：{e}")
            return ""

    # -----------------------------------------------------------
    # Prompts
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
            gen_btn = st.button("🌟 生成靈感樹", width='stretch')
        with col2:
            clr_btn = st.button("🧹 清空", width='stretch')

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
        st.session_state.keyword = keyword.strip()
        with st.spinner("Gemini 正在生成靈感樹..."):
            text = gemini_generate(BASE_PROMPT.format(keyword=keyword))
        data = parse_json_loose(text)
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

    def render_node(node: Dict[str, Any], level=0, idx_path="0", parent_titles = None):
        pad = 20 * level
        parent_titles = parent_titles or []

        with st.container():
            # open div for parent
            st.markdown(f"<div style='margin-left:{pad}px'></div>", unsafe_allow_html=True)

            # --- expander for this node ---
            is_open = (st.session_state.open_path and idx_path.startswith(st.session_state.open_path)) or (level == 0)
            with st.expander(f"▾ **{node['title']}**", expanded=is_open):
                st.write(node.get("desc", "無描述"))

                deep_col = st.text_input(
                    "💡 想深入探討什麼？（可留空使用預設）",
                    key=f"deep_input_{idx_path}",
                    placeholder="例如：永續包裝 / 新口味創新"
                )

                cols = st.columns([1, 1, 1, 4])
                with cols[0]:
                    if st.button("➕ 深入", key=f"expand_{idx_path}"):
                        with st.spinner("延伸子靈感..."):
                            deeper_topic = deep_col.strip()
                            text = gemini_generate(EXPAND_PROMPT.format(
                                keyword=st.session_state.keyword,
                                deep_dive=deeper_topic or "（無特定方向）",
                                title=node["title"],
                                desc=node.get("desc", ""),
                                context=" > ".join(parent_titles + [node["title"]])
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

                # ✅ render children INSIDE the expander and parent container
                for j, child in enumerate(node.get("children", [])):
                    render_node(child, level + 1, f"{idx_path}.{j}", parent_titles + [node["title"]])

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


# ===========================================================
# TAB 2: Deeper Research AI (Trainable / Regulation)
# ===========================================================
with tab2:
    st.header("🔬 深度研發與法規 AI")
    st.caption("用於分析食品材料、製作方法與不同地區食品法規。")

    st.markdown("""
    這個 AI 模型專注於：
    - 食品原料組成與工藝流程；
    - 不同地區的法規與添加物標準；
    - 可持續材料與食品安全合規分析。
    
    （你可以先訓練此模型，然後在此頁面連線使用。）
    """)

    # Links to training or data resources
    st.markdown("### 📚 訓練資料與參考來源")
    st.markdown("""
    **📚 主要參考法規資料來源：**
    - 🇲🇴 [澳門食品安全中心 - 食品添加劑資料](https://www.iam.gov.mo/foodsafety/c/lawstandard/list) ✅  
    - 🇭🇰 [香港食物安全中心 - 食品添加劑守則](https://www.cfs.gov.hk/tc_chi/food_leg/food_leg.html) ✅  

    🌏 其他參考（尚未確認官方連結）：
    -  🇹🇼 [台灣食品添加物法規](https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=L0040001) 
    - 🇯🇵 [日本食品衛生法 (Food Sanitation Act)](https://www.mhlw.go.jp/stf/seisakunitsuite/bunya/0000066597.html) 
    - 🇨🇳 [中國食品添加劑使用標準 GB 2760](https://www.samr.gov.cn/)
    - 🇰🇷 [韓國食品添加物法規 (MFDS)](https://www.mfds.go.kr/)
    - 🇸🇬 [新加坡食品法規 (SFA)](https://www.sfa.gov.sg/food-information/food-labelling/food-additives)
    - 🇪🇺 [EU Food Additives Database](https://webgate.ec.europa.eu/foods_system/main/?sector=FAD)
    - 🇺🇸 [US FDA Food Ingredients Regulations](https://www.fda.gov/food/food-additives-petitions/food-additive-status-list)
    """)


    all_asian_countries = ["台灣", "日本", "中國", "韓國", "新加坡", "泰國", "越南", "香港", "澳門"]
    selected_countries = st.multiselect(
        "🌏 選擇要查詢的地區（可多選）",
        options=all_asian_countries,
        default=["台灣"]
    )

    # 主要的法規網站連結
    country_links = {
        "台灣": "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=L0040001",  # need to fix
        "日本": "https://www.mhlw.go.jp/stf/seisakunitsuite/bunya/0000066597.html", # need to fix
        "中國": "https://www.samr.gov.cn/", # need to fix
        "香港": "https://www.cfs.gov.hk/tc_chi/food_leg/food_leg.html",
        "澳門": "https://www.iam.gov.mo/foodsafety/c/lawstandard/list"
    }
    
    st.divider()
    # Input and send to your external AI site
    st.subheader("🌎 法規與食材分析查詢")
    st.markdown("輸入或貼上要分析的食品概念。AI 將在外部系統中執行深度分析。")
   

    if "concept_input" not in st.session_state:
        st.session_state.concept_input = ""

    concept_input = st.text_area(
        "🧾 想分析的概念 / 食品說明",
        value=st.session_state.concept_input,
        placeholder="例如：抹茶流心大福 / 紫薯拿鐵 / 便利店甜點組合包",
    )

    if st.button("🔍 分析食譜與各地法規", width='stretch'):
        if not concept_input.strip():
            st.warning("請先輸入要分析的概念。")
        else:
            with st.spinner("AI 正在生成食譜與查詢各地法規..."):

                # 1️⃣ Generate or expand recipe if missing
                recipe_prompt = f"""
                你是一位亞洲食品量產研發專家。根據以下描述「{concept_input}」：
                - 若內容未含食譜，請生成一份詳細食譜（含主要成份與添加物）。
                - 若內容已有食譜，請萃取所有食材與添加物。
                - 若可能，請補充簡單的製作步驟，並標註哪些步驟在量產時需特別注意（例如：加熱、保存、攪拌溫度、設備要求）。

                請務必輸出以下格式的 JSON：
                {{
                "食譜名稱": "...",
                "主要食材": ["..."],
                "添加物": ["..."],
                "製作步驟": ["..."],
                "量產重點": ["..."]
                }}
                """
                recipe_text = gemini_generate(recipe_prompt)
                recipe_json = parse_json_loose(recipe_text)

                recipe_json = parse_json_loose(gemini_generate(recipe_prompt))
                additives = recipe_json.get("添加物") or recipe_json.get("additives", [])
                ingredients = recipe_json.get("主要食材") or recipe_json.get("ingredients", [])
                method = recipe_json.get("製作步驟") or recipe_json.get("method", [])
                key_points = recipe_json.get("量產重點", [])
                st.write("🍱 食譜名稱：", recipe_json.get("食譜名稱", "未命名食譜"))
                st.write("🥬 主要食材：", ", ".join(ingredients) if ingredients else "（無）")
                st.write("🧂 添加物：", ", ".join(additives) if additives else "（無）")
                if method:
                    st.write("🍳 **製作步驟**：")
                    for i, step in enumerate(method, start=1):
                        st.markdown(f"{i}. {step}")
                if key_points:
                    st.write("🏭 **量產重點**：")
                    for k in key_points:
                        st.markdown(f"- {k}")

                if not additives:
                    st.info("未偵測到特定添加物，AI 將僅顯示主要食材資訊。")

                # 2️⃣ Build a regulation table
                reg_results = []
                for country in selected_countries:
                    for add in additives:
                        reg_prompt = f"""
                        你可以使用 Google Search 進行查詢。
                        請在以下網站中搜尋「{add}」在「{country}」的食品添加物法規資訊：
                        {country_links[country]}

                        請回答：
                        - 是否允許使用
                        - 最大添加量
                        - 適用食品類別
                        - 提供官方來源網址

                        以 JSON 格式輸出：
                        {{
                        "國家": "{country}",
                        "添加物": "{add}",
                        "使用狀態": "...",
                        "最大添加量": "...",
                        "適用食品類別": "...",
                        "官方來源": "..."
                        }}
                        """
                        reg_json = parse_json_loose(gemini_generate(reg_prompt))
                        reg_results.append(reg_json)
                try:
                    df = pd.DataFrame(reg_results)
                    # 確保欄位順序一致
                    cols = ["國家", "添加物", "使用狀態", "最大添加量", "適用食品類別", "官方來源"]
                    df = df[[c for c in cols if c in df.columns]]
                    st.dataframe(df, width='stretch')
                    st.success("✅ 完成法規查詢（由 Gemini + Google Search 生成）")
                except Exception as e:
                    st.error(f"無法解析 Gemini 輸出：{e}")
    else:
        st.warning("未能找到相關法規資訊，請檢查輸入或稍後再試。")