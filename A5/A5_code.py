import streamlit as st
import spacy
import requests
import re

# ================= 全局页面配置 =================
st.set_page_config(page_title="深层语义与篇章分析平台", layout="wide", initial_sidebar_state="expanded")

# ================= 全局高级浅色 CSS 注入 =================
css = """
<style>
/* 全局浅色高级渐变背景 */
.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
}
/* 标题渐变色与阴影特效 (沉稳的深空蓝紫) */
h1 {
    color: #2b5876 !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
    padding-bottom: 5px;
}
h2, h3, h4 {
    color: #2c3e50;
    font-weight: 700;
}
/* 左侧边栏毛玻璃材质与提亮 */
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.4) !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.6);
    box-shadow: 2px 0 15px rgba(0,0,0,0.03);
}
/* 隐藏顶部红线 */
header[data-testid="stHeader"] {
    background: transparent;
}
/* 魔改 Primary Button（激活态长框）使其变为沉浮、柔和的灰蓝深蓝渐变色，避免荧光感扎眼 */
div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 10px rgba(24, 40, 72, 0.2) !important;
    border-radius: 12px !important;
    height: 54px !important;
    font-weight: 700 !important;
    font-size: 1.05em !important;
    transition: all 0.3s ease !important;
    margin-bottom: 5px;
}
div.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 14px rgba(24, 40, 72, 0.35) !important;
}
/* 魔改 Secondary Button（未激活长框） */
div.stButton > button[kind="secondary"] {
    background: rgba(255, 255, 255, 0.6) !important;
    color: #555 !important;
    border: 1px solid rgba(255, 255, 255, 0.8) !important;
    border-radius: 12px !important;
    height: 54px !important;
    font-weight: 600 !important;
    font-size: 1.05em !important;
    box-shadow: 0 2px 5px rgba(0,0,0,0.02) !important;
    transition: all 0.3s ease !important;
    margin-bottom: 5px;
}
div.stButton > button[kind="secondary"]:hover {
    transform: translateY(-2px) !important;
    background: rgba(255, 255, 255, 0.95) !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05) !important;
    color: #111 !important;
}
/* 自定义区块的圆角白底泛光化 */
.streamlit-expanderHeader {
    border-radius: 10px;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)


# ================= 状态路由管理 =================
PAGE_1 = "✂️ 模块 1：细粒度话语分割 (EDU)"
PAGE_2 = "🔗 模块 2：浅层篇章分析与连接词"
PAGE_3 = "🧬 模块 3：端到端指代消解聚类"

if "current_page" not in st.session_state:
    st.session_state.current_page = PAGE_1

# ================= 侧边栏构建 =================
with st.sidebar:
    st.markdown("<h2 style='text-align: center; margin-top: -10px; font-size: 1.8em;'>🧭 探索导航</h2>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.15), rgba(0, 0, 0, 0)); margin: 10px 0 20px 0;'>", unsafe_allow_html=True)
    
    # 模拟状态监控改变颜色
    if st.button(PAGE_1, use_container_width=True, type="primary" if st.session_state.current_page == PAGE_1 else "secondary"):
        st.session_state.current_page = PAGE_1
        st.rerun()
        
    if st.button(PAGE_2, use_container_width=True, type="primary" if st.session_state.current_page == PAGE_2 else "secondary"):
        st.session_state.current_page = PAGE_2
        st.rerun()
        
    if st.button(PAGE_3, use_container_width=True, type="primary" if st.session_state.current_page == PAGE_3 else "secondary"):
        st.session_state.current_page = PAGE_3
        st.rerun()

# ================= 主标题展示 =================
st.title("✨ 深层语义与篇章分析平台")
st.subheader("Week 6: Discourse & Coreference")
st.markdown("<br>", unsafe_allow_html=True)


# ================= 全局核心功能依赖 =================
@st.cache_resource
def load_spacy_model():
    """复用 spaCy 模型"""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("未能找到 spaCy 模型 'en_core_web_sm'，请在终端执行: `python -m spacy download en_core_web_sm`")
        return None

@st.cache_data
def fetch_ground_truth_edus(url):
    """根据指定的 URL 获取 NeuralEDUSeg 的 EDU Ground Truth 文件的内容"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        text_data = response.text
        return [line.strip() for line in text_data.strip().split('\n') if line.strip()]
    except Exception as e:
        st.error(f"⚠️ 获取数据失败或发生网络异常: {e}")
        return []

def segment_edu_rule_based(text, nlp):
    """基于基线规则利用依存句法分析切分 EDU"""
    doc = nlp(text)
    edus = []
    
    for sent in doc.sents:
        current_edu_tokens = []
        for token in sent:
            if token.pos_ == "SCONJ" and len(current_edu_tokens) > 0:
                edus.append("".join(current_edu_tokens).strip())
                current_edu_tokens = []
            
            current_edu_tokens.append(token.text_with_ws)
            
            is_punct_break = token.text in [',', ';'] and token.i < sent.end - 1
            is_sent_end = token.i == sent.end - 1
            
            if is_punct_break or is_sent_end:
                edus.append("".join(current_edu_tokens).strip())
                current_edu_tokens = []
                
        if current_edu_tokens:
            edus.append("".join(current_edu_tokens).strip())
            
    return [e for e in edus if len(e.strip()) > 0]

def render_edus_html(edu_list):
    """利用 HTML 与 CSS 样式渲染卡片式 EDU 列表"""
    html_content = ""
    for edu in edu_list:
        parts = edu.split()
        if not parts: continue
        
        last_word = parts[-1]
        highlighted_word_html = f'<span style="background-color: #ffe066; color: #000; padding: 2px 4px; border-radius: 4px; font-weight: bold; margin-left: 2px; box-shadow: 0 1px 2px rgba(0,0,0,0.1);" title="Boundary Token">{last_word}</span>'
        parts[-1] = highlighted_word_html
        edu_html_str = " ".join(parts)
        
        # 将左侧标识带修改为深沉护眼的灰蓝色 (#4b6cb7)
        card_html = f'''
        <div style="background-color: rgba(255, 255, 255, 0.85); border: 1px solid rgba(0,0,0,0.05); border-left: 5px solid #4b6cb7; border-radius: 8px; padding: 12px 18px; margin-bottom: 16px; font-family: Arial, sans-serif; color: #333; box-shadow: 0 2px 6px rgba(0,0,0,0.03); line-height: 1.6;">
            {edu_html_str}
        </div>
        '''
        html_content += card_html
    return html_content

@st.cache_resource
def load_coref_model():
    """按需加载 fastcoref 模型，避免重复实例化"""
    try:
        from fastcoref import FCoref
        return FCoref(device='cpu')
    except ImportError:
        st.error("无法加载 fastcoref，请在终端执行: `pip install fastcoref`")
        return None
    except Exception as e:
        st.error(f"加载 fastcoref 时发生异常: {e}")
        return None


# ================= 业务路由分发 =================

# ----------------- 页面一：话语分割 -----------------
if st.session_state.current_page == PAGE_1:
    st.markdown("### 基于规则与标注的话语分割比对")
    
    with st.container():
        run_btn = st.button("🚀 运行 EDU 切分对比", use_container_width=True)
        
    if run_btn:
        nlp = load_spacy_model()
        if nlp is not None:
            with st.spinner("🌍 正在加载模型及抓取原始数据中..."):
                url = "https://raw.githubusercontent.com/PKU-TANGENT/NeuralEDUSeg/master/data/rst/TRAINING/wsj_0601.out.edus"
                ground_truth_edus = fetch_ground_truth_edus(url)
                
            if ground_truth_edus:
                orig_text = " ".join(ground_truth_edus)
                rule_edus = segment_edu_rule_based(orig_text, nlp)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### ⚙️ 规则基线切分 (spaCy)")
                    st.markdown(render_edus_html(rule_edus), unsafe_allow_html=True)
                with col2:
                    st.markdown("#### 🧠 真实标注 (Ground Truth)")
                    st.markdown(render_edus_html(ground_truth_edus), unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("📊 模型机制解析与性能比对 (默认展开)", expanded=True):
                    analysis_html1 = '''
                    <div style="display: flex; gap: 20px; align-items: flex-start; justify-content: space-between;">
                        <div style="flex: 1; background: rgba(255,255,255,0.7); padding: 15px; border-radius: 8px; border: 1px solid #dce1e6;">
                            <h4 style="margin-top: 0; color: #2c3e50;">⚙️ 启发式基线 (spaCy Rules)</h4>
                            <p style="font-size: 0.9em; color: #555;">利用 dependency parser 定位从句引导词并强行打断。优势在于免显卡、极其轻量。</p>
                            <div style="margin-bottom: 8px;">
                                <div style="display: flex; justify-content: space-between; font-size: 0.85em;"><span>推断速度 (Speed)</span><span>⚡ 极快</span></div>
                                <div style="width: 100%; background-color: #eee; border-radius: 4px; overflow: hidden;"><div style="width: 95%; height: 8px; background-color: #4CAF50;"></div></div>
                            </div>
                            <div>
                                <div style="display: flex; justify-content: space-between; font-size: 0.85em;"><span>复杂从句容错率 (Accuracy)</span><span>❌ 较差</span></div>
                                <div style="width: 100%; background-color: #eee; border-radius: 4px; overflow: hidden;"><div style="width: 40%; height: 8px; background-color: #F44336;"></div></div>
                            </div>
                        </div>
                        <div style="flex: 1; background: rgba(255,255,255,0.7); padding: 15px; border-radius: 8px; border: 1px solid #dce1e6; border-left: 4px solid #1f77b4;">
                            <h4 style="margin-top: 0; color: #2c3e50;">🧠 NeuralEDUSeg (深度学习)</h4>
                            <p style="font-size: 0.9em; color: #555;">引入上下文感知编码器 (BiLSTM 等)，能学习整句话结构分布，抗长句干扰能力极强。</p>
                            <div style="margin-bottom: 8px;">
                                <div style="display: flex; justify-content: space-between; font-size: 0.85em;"><span>推断速度 (Speed)</span><span>⏳ 偏慢</span></div>
                                <div style="width: 100%; background-color: #eee; border-radius: 4px; overflow: hidden;"><div style="width: 55%; height: 8px; background-color: #FF9800;"></div></div>
                            </div>
                            <div>
                                <div style="display: flex; justify-content: space-between; font-size: 0.85em;"><span>跨句理解泛化度 (Accuracy)</span><span>✅ 极佳</span></div>
                                <div style="width: 100%; background-color: #eee; border-radius: 4px; overflow: hidden;"><div style="width: 92%; height: 8px; background-color: #4CAF50;"></div></div>
                            </div>
                        </div>
                    </div>
                    '''
                    st.markdown(analysis_html1, unsafe_allow_html=True)

# ----------------- 页面二：显式关系提取 -----------------
elif st.session_state.current_page == PAGE_2:
    st.markdown("### 基于 PDTB 理论的显式关系浅层抽提")
    
    pdtb_dict = {
        "TEMPORAL": ["when", "after", "before", "then", "since"],
        "CONTINGENCY": ["because", "if", "so", "thus", "therefore", "since"],
        "COMPARISON": ["but", "however", "although", "though", "nevertheless"],
        "EXPANSION": ["and", "or", "moreover", "furthermore", "also"]
    }
    
    default_text = "Third-quarter sales in Europe were exceptionally strong, boosted by promotional programs and new products - although weaker foreign currencies reduced the company's earnings."
    user_text = st.text_area("输入待分析的长句：", value=default_text, height=120)
    
    if st.button("🔍 提取篇章论元 (Extract Arguments)", use_container_width=True):
        found_match = None
        earliest_pos = len(user_text)
        
        for pdtb_class, connectives in pdtb_dict.items():
            for conn in connectives:
                pattern = r'\b' + re.escape(conn) + r'\b'
                match = re.search(pattern, user_text, re.IGNORECASE)
                if match and match.start() < earliest_pos:
                    earliest_pos = match.start()
                    found_match = {"conn_text": match.group(), "start": match.start(), "end": match.end(), "class": pdtb_class}
        
        if found_match:
            arg1 = user_text[:found_match['start']].rstrip()
            arg2 = user_text[found_match['end']:].lstrip()
            conn_display = f"{found_match['conn_text'].lower()} [{found_match['class']}]"
            
            # 使用较深的经典科技灰（而不是突兀的霓虹色）作为对比
            html_result = f'''
            <div style="font-size: 1.1em; line-height: 1.8; padding: 20px; border-radius: 12px; background-color: rgba(255, 255, 255, 0.85); box-shadow: 0 4px 10px rgba(0,0,0,0.03); margin-top: 10px;">
                <span style="background-color: #e3f2fd; color: #0d47a1; padding: 4px 8px; border-radius: 4px;">{arg1}</span>
                <span style="color:white; background: linear-gradient(135deg, #1f1c2c 0%, #444055 100%); padding: 4px 10px; border-radius:4px; font-weight: bold; margin: 0 6px;">{conn_display}</span>
                <span style="background-color: #fff9c4; color: #f57f17; padding: 4px 8px; border-radius: 4px;">{arg2}</span>
            </div>
            '''
            st.markdown("#### 🧠 结构化论元提取结果")
            st.markdown(html_result, unsafe_allow_html=True)
        else:
            st.info("💡 未检测到预设的显式篇章连接词。")
            
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📊 算法机理解剖：正则提取 vs 语义树分析 (默认展开)", expanded=True):
            analysis_html2 = '''
            <div style="background: rgba(255,255,255,0.7); border: 1px solid #dce1e6; border-radius: 8px; padding: 15px;">
                <h4 style="margin-top: 0; color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 8px;"> PDTB 结构化解析痛点与破局</h4>
                <div style="display: flex; align-items: center; justify-content: space-around; margin-top: 15px;">
                    <div style="text-align: center; max-width: 30%;">
                        <div style="font-size: 2.5em; margin-bottom: 10px;">🔍</div>
                        <strong style="color: #333;">浅层词典匹配</strong>
                        <p style="font-size: 0.85em; color: #666; margin-top: 5px;">如提取 "since" 时，硬规则无法判断它是【Temporal时间】还是【Contingency因果】，极度缺乏消歧能力。</p>
                    </div>
                    <div style="font-size: 1.5em; color: #ccc;">➔</div>
                    <div style="text-align: center; max-width: 30%;">
                        <div style="font-size: 2.5em; margin-bottom: 10px;">🧩</div>
                        <strong style="color: #333;">隐式关系探测</strong>
                        <p style="font-size: 0.85em; color: #666; margin-top: 5px;">语料库中常有近半数语句不带任何连接词。规则引擎此地会直接失效，需要语言模型通过语义向量测算来补齐意图。</p>
                    </div>
                    <div style="font-size: 1.5em; color: #ccc;">➔</div>
                    <div style="text-align: center; max-width: 30%;">
                        <div style="font-size: 2.5em; margin-bottom: 10px;">🌲</div>
                        <strong style="color: #333;">深层句法树分析网</strong>
                        <p style="font-size: 0.85em; color: #666; margin-top: 5px;">利用 Biaffine Parser 将文本建模为图结构，进行长距离词依附，能完美破译 Arg1、Arg2 的远距离跨越或倒装难题。</p>
                    </div>
                </div>
            </div>
            '''
            st.markdown(analysis_html2, unsafe_allow_html=True)

# ----------------- 页面三：指代消解 -----------------
elif st.session_state.current_page == PAGE_3:
    st.markdown("### 端到端神经网络指代消解视图")
    
    default_coref_text = "Barack Obama is an American politician who served as the 44th president of the United States. He was the first African-American president. His historic presidency began in 2009."
    coref_text = st.text_area("输入待分析的实体代词文本：", value=default_coref_text, height=150)
    
    if st.button("🧬 运行全局指代聚类 (Run Coref)", use_container_width=True):
        coref_model = load_coref_model()
        if coref_model is not None:
            with st.spinner("🧠 正在进行神经网络推理与聚类，请稍候..."):
                try:
                    preds = coref_model.predict(texts=[coref_text])
                    pred_result = preds[0]
                    clusters_indices = pred_result.get_clusters(as_strings=False)
                    
                    if not clusters_indices:
                        st.info("💡 未检测到共指集群。")
                    else:
                        # 换回学术低对比度浅色护眼柔和调色板
                        color_palette = ['#ffcdd2', '#bbdefb', '#c8e6c9', '#ffe082', '#e1bee7', '#f8bbd0', '#b2ebf2', '#dcedc8', '#ffcc80', '#cfd8dc']
                        all_mentions = []
                        for c_idx, c_spans in enumerate(clusters_indices):
                            for start, end in c_spans:
                                all_mentions.append({"start": start, "end": end, "cluster_idx": c_idx, "bg": color_palette[c_idx % len(color_palette)]})
                                
                        all_mentions.sort(key=lambda x: x["start"], reverse=True)
                        
                        highlighted_text = coref_text
                        for m in all_mentions:
                            span_html = f'<span style="background-color: {m["bg"]}; border-radius: 4px; padding: 2px 4px; margin: 0 2px; color: #333; font-weight: 500;">{highlighted_text[m["start"]:m["end"]]} <sup style="color: #666;">[C{m["cluster_idx"]+1}]</sup></span>'
                            highlighted_text = highlighted_text[:m["start"]] + span_html + highlighted_text[m["end"]:]
                            
                        st.markdown("#### ✨ 原文指代高亮大区")
                        st.markdown(f'<div style="font-size: 1.15em; line-height: 2; padding: 20px; border-radius: 12px; background-color: rgba(255, 255, 255, 0.85); box-shadow: 0 4px 10px rgba(0,0,0,0.03); margin-top: 10px; margin-bottom: 25px;">{highlighted_text}</div>', unsafe_allow_html=True)
                        
                        st.markdown("#### 🗂️ 结构化等价类簇")
                        for c_idx, cluster_strs in enumerate(pred_result.get_clusters(as_strings=True)):
                            unique_strs = []
                            for s in cluster_strs:
                                if s not in unique_strs: unique_strs.append(s)
                            st.markdown(f'''
                            <div style="margin-bottom: 12px; font-size: 1.05em; background: rgba(255, 255, 255, 0.9); padding: 12px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.03);">
                                <span style="display:inline-block; width: 16px; height: 16px; background-color: {color_palette[c_idx % len(color_palette)]}; border-radius: 4px; margin-right: 10px; vertical-align: text-bottom;"></span>
                                <strong>Cluster {c_idx+1}:</strong> 
                                <span style="color: #444; font-family: Consolas, monospace;">['{"', '".join(unique_strs)}']</span>
                            </div>
                            ''', unsafe_allow_html=True)
                            
                        st.markdown("<br>", unsafe_allow_html=True)
                        with st.expander("📈 指代消解算法演变史与代价比对 (默认展开)", expanded=True):
                            analysis_html3 = '''
                            <div style="background: rgba(255,255,255,0.7); border-radius: 8px; border: 1px solid #e1e4e8; padding: 16px;">
                                <h4 style="margin-top: 0; color: #1e293b;">🧬 Coref 算法演化天梯</h4>
                                <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 0.95em;">
                                    <tr style="border-bottom: 2px solid #cbd5e1; color: #475569;">
                                        <th style="padding: 10px;">时代及算法代表</th>
                                        <th style="padding: 10px;">核心运算机制</th>
                                        <th style="padding: 10px;">跨段落推理 F1</th>
                                        <th style="padding: 10px;">计算代价与落脚点</th>
                                    </tr>
                                    <tr style="border-bottom: 1px dashed #e2e8f0; background: #f8fafc;">
                                        <td style="padding: 10px; font-weight: 500;">Hobbs (1970s)</td>
                                        <td style="padding: 10px;">从右向左暴力遍历 Penn Treebank 句法树，寻找先行词</td>
                                        <td style="padding: 10px; color: #ef4444;">极低 (~ 50%)</td>
                                        <td style="padding: 10px;"><span style="background:#dcfce7; color:#166534; padding:2px 6px; border-radius:4px; font-size:0.9em;">算力低</span> 但极其依赖纯净无倒错的语法库</td>
                                    </tr>
                                    <tr style="border-bottom: 1px dashed #e2e8f0;">
                                        <td style="padding: 10px; font-weight: 500;">Mention-Pair Model</td>
                                        <td style="padding: 10px;">利用机器学习监督特征，强行将任意词块配对做二元评估</td>
                                        <td style="padding: 10px; color: #f59e0b;">一般 (~ 65%)</td>
                                        <td style="padding: 10px;"><span style="background:#fef9c3; color:#854d0e; padding:2px 6px; border-radius:4px; font-size:0.9em;">成本适中</span> 极易生成逻辑相悖的错误环路</td>
                                    </tr>
                                    <tr style="background: rgba(240, 253, 244, 0.6);">
                                        <td style="padding: 10px; font-weight: 600; color: #166534;">Neural End-to-End</td>
                                        <td style="padding: 10px;">Transformer 稠密向量动态高维投影，全局图直接聚类</td>
                                        <td style="padding: 10px; color: #15803d; font-weight: bold;">顶尖 (> 80%)</td>
                                        <td style="padding: 10px;"><span style="background:#fee2e2; color:#991b1b; padding:2px 6px; border-radius:4px; font-size:0.9em;">耗显存</span> 需要强大算力和极大参数模型</td>
                                    </tr>
                                </table>
                            </div>
                            '''
                            st.markdown(analysis_html3, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"发生错误：{e}")
