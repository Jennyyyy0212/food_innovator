import streamlit as st
from google import genai
from streamlit_local_storage import LocalStorage

from modules.ai_innovation import render_innovation
from modules.ai_receipt import render_receipt
from modules.ai_research import render_research
from modules.ai_favorites import render_favorites
from gemini_utils import gemini_generate

# Must be first Streamlit call
st.set_page_config(
    page_title="AI 食品靈感引擎 (Gemini + Streamlit)",
    page_icon="🌿",
    layout="wide"
)

# Instantiate LocalStorage before any st.stop() so the hidden component always
# renders and localStorage remains accessible regardless of auth state.
st.session_state["_ls"] = LocalStorage()

# -----------------------------------------------------------
# Shared sidebar – API key, model, test connection
# -----------------------------------------------------------
with st.sidebar:
    st.header("🔐 API 設定")
    api_key = st.text_input(
        "Gemini API 金鑰",
        type="password",
        placeholder="輸入你的 Google API 金鑰",
        key="api_key_input",
    )
    if api_key:
        st.success("✅ 已設定 Gemini API 金鑰")
    else:
        st.warning("⚠️ 請輸入 API 金鑰")

    model_name = st.selectbox(
        "Gemini 模型",
        [
            "gemini-flash-latest",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
        ],
        index=0,
    )
    st.markdown("---")
    st.caption("到 Google AI Studio 取得金鑰 → https://aistudio.google.com/")

if not api_key:
    st.stop()

try:
    client = genai.Client(api_key=api_key)
except Exception as e:
    st.error(f"❌ 初始化 Gemini 失敗：{e}")
    st.stop()

with st.sidebar:
    st.markdown("---")
    st.subheader("🧪 測試連線")
    if st.button("🔍 測試 Gemini API 連線"):
        result = gemini_generate(
            client,
            model_name,
            "請用一句話描述亞洲便利店飲品創新趨勢。",
            on_retry=lambda attempt, delay: st.toast(
                f"⏳ Gemini 暫時繁忙，{delay} 秒後重試（第 {attempt + 1} 次）…"
            ),
            on_error=lambda e: st.error(f"❌ Gemini 錯誤：{e}"),
        )
        if result:
            st.success("✅ 連線成功！Gemini 回覆：")
            st.write(result)

# -----------------------------------------------------------
# Tabs
# -----------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "💡 食品靈感生成 AI",
    "🧾 食譜生成 AI",
    "🔬 深度研發與法規 AI",
    "⭐ 收藏清單",
])

with tab1:
    render_innovation(client, model_name)

with tab2:
    render_receipt(client, model_name)

with tab3:
    render_research(client, model_name)

with tab4:
    render_favorites(client, model_name)
