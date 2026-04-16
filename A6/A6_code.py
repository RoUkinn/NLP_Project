import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import time
import nltk
from transformers import pipeline
from nltk.tokenize import word_tokenize
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends
from nltk.lm import MLE, Laplace
from nltk.util import ngrams

# ==================== 环境与数据初始化 ====================
@st.cache_resource
def setup_env():
    """
    检查并下载必需的 nltk 分词依赖补丁。
    采用静默下载机制防止打断应用启动流程。
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
        
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)

setup_env()

# ==================== 全局页面配置 ====================
st.set_page_config(page_title="语言模型训练与对比分析平台", layout="wide", initial_sidebar_state="expanded")

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
/* 魔改 Primary Button（激活态长框） */
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
PAGE_1 = "📊 模块 1：n 元语模型与数据平滑"
PAGE_2 = "🔥 模块 2：从零训练 RNN 语言模型"
PAGE_3 = "🆚 模块 3：预训练架构对比"
PAGE_4 = "🧠 模块 4：语言模型评价 (Perplexity)"

if "current_page" not in st.session_state:
    st.session_state.current_page = PAGE_1

# ================= 侧边栏构建 =================
with st.sidebar:
    st.markdown("<h2 style='text-align: center; margin-top: -10px; font-size: 1.8em;'>🧭 探索导航</h2>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.15), rgba(0, 0, 0, 0)); margin: 10px 0 20px 0;'>", unsafe_allow_html=True)
    
    if st.button(PAGE_1, use_container_width=True, type="primary" if st.session_state.current_page == PAGE_1 else "secondary"):
        st.session_state.current_page = PAGE_1
        st.rerun()
        
    if st.button(PAGE_2, use_container_width=True, type="primary" if st.session_state.current_page == PAGE_2 else "secondary"):
        st.session_state.current_page = PAGE_2
        st.rerun()
        
    if st.button(PAGE_3, use_container_width=True, type="primary" if st.session_state.current_page == PAGE_3 else "secondary"):
        st.session_state.current_page = PAGE_3
        st.rerun()

    if st.button(PAGE_4, use_container_width=True, type="primary" if st.session_state.current_page == PAGE_4 else "secondary"):
        st.session_state.current_page = PAGE_4
        st.rerun()


# ================= 主标题展示 =================
st.title("✨ 语言模型训练与对比分析平台")
st.subheader("Week 7: 从 n-gram 到大语言模型的基石演练")
st.markdown("<br>", unsafe_allow_html=True)

default_corpus = """
The rapid advancement of artificial intelligence has transformed the tech industry. 
The company announced new products that utilize deep learning to improve efficiency. 
Many developers are now focusing on natural language processing applications. 
Machine learning models require huge amounts of training data, and the company has invested heavily in data collection.
Our new framework makes it easier to train neural networks.
We expect these technologies to become ubiquitous in the near future.
"""

@st.cache_resource
def train_models(corpus_text, n=3):
    tokens = word_tokenize(corpus_text.lower())
    
    train_data_mle, vocab_mle = padded_everygram_pipeline(n, [tokens])
    mle_model = MLE(n)
    mle_model.fit(train_data_mle, vocab_mle)
    
    train_data_lap, vocab_lap = padded_everygram_pipeline(n, [tokens])
    laplace_model = Laplace(n)
    laplace_model.fit(train_data_lap, vocab_lap)
    
    return mle_model, laplace_model

# ==================== 业务路由分发 ====================

if st.session_state.current_page == PAGE_1:
    col_input, col_info = st.columns([2, 1])
    
    with col_info:
        st.info("💡 **什么是语言模型 (LM)?** \n\n语言模型负责计算一段特征语句的成立概率。它是判定机器生成语句是“像人话”还是“胡言乱语”的核心判卷工具。")
        
    with col_input:
        st.markdown("### 📚 1. 训练数据配置区")
        # 让高管/测试人员输入训练文本（系统自动将其消化为底层词表 Vocabulary）
        corpus = st.text_area("上传训练语料库 (Training Corpus)：", value=default_corpus.strip(), height=150)
    
    if corpus:
        # 当数据存在，立刻发起后端模型缓存与训练
        mle_model, laplace_model = train_models(corpus, n=3)
        
        st.markdown("---")
        st.markdown("### 🧠 2. 条件概率链式打分与平滑控制")
        
        st.markdown("在这个试验台中，您可以任意向一个已学习过上述语料集的 Trigram (三元模型) 提出假设句子，观察它判定该句子的**联合出现概率**有多大。")
        
        input_col, toggle_col = st.columns([3, 1])
        with input_col:
            # “smartphones” 作为完全不存在于语料集中的 OOV 词，将被用作零概率地雷弹
            test_sentence = st.text_input("注入一条测试探针句子：", value="The company announced new smartphones.")
        with toggle_col:
            st.markdown("<br>", unsafe_allow_html=True)
            # 通过拨动开关让模型瞬间切换
            use_laplace = st.toggle("开启加一平滑 (Laplace Smoothing)", value=True)
            
        current_model = laplace_model if use_laplace else mle_model
        
        # 抛出学术公式装点版面
        with st.expander("📖 核心底层概率公式与分布定理 (默认展开)", expanded=True):
            if use_laplace:
                st.latex(r"P(w_i | w_{i-2}, w_{i-1}) = \frac{C(w_{i-2}, w_{i-1}, w_i) + 1}{C(w_{i-2}, w_{i-1}) + V}")
                st.caption("✨ **Laplace 保护机制已激活**：分子的微小增量相当于给系统中每个可能的随机事件塞了张底层福利彩票，使得未知词条或组合获得了极少但绝不为 `0` 的概率分配。")
            else:
                st.latex(r"P(w_i | w_{i-2}, w_{i-1}) = \frac{C(w_{i-2}, w_{i-1}, w_i)}{C(w_{i-2}, w_{i-1})}")
                st.caption("⚙️ **原始大自然选择(MLE)计算中**：该机制苛责要求数据严格忠诚于被训词条统计特征。不认识的跨度词对？一律判定为绝对不可能事件(`0`)。")

        if test_sentence:
            st.markdown("#### 📊 Trigram 组件解码分析台")
            
            # 使用分词器切解测试探针
            test_tokens = word_tokenize(test_sentence.lower())
            
            # 使用包含头尾填充句柄 (<s> </s>) 的模式拼装 Trigram
            padded_test = list(pad_both_ends(test_tokens, n=3))
            test_trigrams = list(ngrams(padded_test, 3))
            
            records = []
            joint_prob = 1.0
            
            for trigram in test_trigrams:
                w3 = trigram[2]
                context = (trigram[0], trigram[1])
                
                # 在 NLTK 的模型中，直接获取内部 count 会得到真正的频数
                count_trigram = mle_model.counts[context][w3]
                
                # 当前 Trigram 在被选中的模型(是否拥有平滑)里经过处理的条件概率预测
                prob = current_model.score(w3, context)
                
                # 联结所有阶段乘积
                joint_prob *= prob
                
                records.append({
                    "词组切片 (Trigrams Chunk)": f"('{context[0]}', '{context[1]}') ➔ '{w3}'",
                    "训练集绝对频数 C(W)": count_trigram,
                    "P条件概率估值 (Conditional Prob)": prob
                })
                
            df = pd.DataFrame(records)
            
            # 使用 Pandas CSS Styling 捕捉零概率地雷并将其高亮为血红色警报
            def highlight_zero_frequency(row):
                if row["训练集绝对频数 C(W)"] == 0:
                    return ['background-color: rgba(239, 68, 68, 0.15); color: #b91c1c; font-weight: bold'] * len(row)
                return [''] * len(row)
                
            st.dataframe(df.style.apply(highlight_zero_frequency, axis=1), use_container_width=True)
            
            st.markdown("#### 🎯 序列句际全局评分 (Joint Probability)")
            
            # 因为概率会被无情缩放到无穷小，因此使用科学计数法防止被直接 UI 省略尾数
            prob_display = f"{joint_prob:e}"
            
            if joint_prob == 0.0:
                st.error("⚠️ **绝对零概率坍塌拦截协议报警！** \n\n游标刚刚检测到了训练集中从未见过的 OOV（未登录词, 见红框分析图层）或全新的 N-gram 语境组合！根据原始 MLE 规则链式的 `0` 会污染整个上游的积结果，导致句级置信度直接清盘。 **方案**：请前往参数控制区打开【Laplace 面板开关】弥补分布缝隙。")
                st.metric("Sentence Final Probability", "0.00e+00")
            else:
                st.success("✅ 测试短句链式测算成功！" + ("平滑池为原本 OOV 元素注入了微弱微元概率，拯救了该句命运边界！" if use_laplace and any(r["训练集绝对频数 C(W)"] == 0 for r in records) else "当前语料特征已悉数捕获验证特征词段。"))
                st.metric("Sentence Final Probability", prob_display)
                
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("📊 模型机制解析与特征打分流派比对 (默认展开)", expanded=True):
                analysis_html_lm = '''
                <div style="display: flex; gap: 20px; align-items: flex-start; justify-content: space-between;">
                    <div style="flex: 1; background: rgba(255,255,255,0.7); padding: 15px; border-radius: 8px; border: 1px solid #dce1e6;">
                        <h4 style="margin-top: 0; color: #2c3e50;">⚙️ 朴素最大似然估计 (MLE)</h4>
                        <p style="font-size: 0.9em; color: #555;">彻底拥抱“只见树木”的数据主义，完全服从现有语料统计特征的基建系统。</p>
                        <div style="margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; font-size: 0.85em;"><span>OOV 容灾泛化性 (Robustness)</span><span>❌ 极弱</span></div>
                            <div style="width: 100%; background-color: #eee; border-radius: 4px; overflow: hidden;"><div style="width: 10%; height: 8px; background-color: #ef4444;"></div></div>
                        </div>
                        <div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.85em;"><span>原本数据特征纯度 (Purity)</span><span>✅ 极佳</span></div>
                            <div style="width: 100%; background-color: #eee; border-radius: 4px; overflow: hidden;"><div style="width: 100%; height: 8px; background-color: #22c55e;"></div></div>
                        </div>
                    </div>
                    <div style="flex: 1; background: rgba(255,255,255,0.7); padding: 15px; border-radius: 8px; border: 1px solid #dce1e6; border-left: 4px solid #1f77b4;">
                        <h4 style="margin-top: 0; color: #2c3e50;">🛡️ 加一平滑协议 (Laplace)</h4>
                        <p style="font-size: 0.9em; color: #555;">在所有分母增加词库库底容量 V，利用人工先验知识给未曾见的事件施予最低生存权。</p>
                        <div style="margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; font-size: 0.85em;"><span>OOV 容灾泛化性 (Robustness)</span><span>✅ 较强</span></div>
                            <div style="width: 100%; background-color: #eee; border-radius: 4px; overflow: hidden;"><div style="width: 85%; height: 8px; background-color: #22c55e;"></div></div>
                        </div>
                        <div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.85em;"><span>原本数据特征纯度 (Purity)</span><span>⏳ 偏低 (产生均态稀释扭曲)</span></div>
                            <div style="width: 100%; background-color: #eee; border-radius: 4px; overflow: hidden;"><div style="width: 45%; height: 8px; background-color: #f59e0b;"></div></div>
                        </div>
                    </div>
                </div>
                '''
                st.markdown(analysis_html_lm, unsafe_allow_html=True)

elif st.session_state.current_page == PAGE_2:
    st.markdown("### 🚂 从零精调字符级 RNN 语言模型")
    
    # === 状态机防御墙：防止组件重组时数据坍塌 ===
    if "is_trained" not in st.session_state:
        st.session_state.is_trained = False
        st.session_state.trained_model = None
        st.session_state.char2int = {}
        st.session_state.int2char = {}
        
    default_poem = "Two roads diverged in a yellow wood,\nAnd sorry I could not travel both\nAnd be one traveler, long I stood\nAnd looked down one as far as I could\nTo where it bent in the undergrowth."
    
    st.markdown("#### 1. 投喂私人语言语料库")
    poem_text = st.text_area("输入用于拟合权重的字符级序列语料 (约200字符体验最佳):", value=default_poem, height=130)
    
    st.markdown("#### 2. 超参数主控台")
    col_hz, col_ep, col_lr = st.columns(3)
    with col_hz:
        hidden_size = st.slider("Hidden Size (高维投影维度)", min_value=16, max_value=128, value=32, step=16)
    with col_ep:
        epochs = st.slider("Epochs (遍历轮数)", min_value=10, max_value=500, value=100, step=10)
    with col_lr:
        lr_options = [0.001, 0.005, 0.01, 0.05]
        lr = st.selectbox("Learning Rate (学习率步长)", lr_options, index=2)
        
    # 定义简约版 PyTorch RNN 模型
    class CharRNN(nn.Module):
        def __init__(self, vocab_size, hidden_size):
            super(CharRNN, self).__init__()
            self.hidden_size = hidden_size
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, vocab_size)
            
        def forward(self, x, hidden):
            embeds = self.embedding(x)
            out, hidden = self.rnn(embeds, hidden)
            out = self.fc(out)
            return out, hidden
            
        def init_hidden(self, batch_size):
            return torch.zeros(1, batch_size, self.hidden_size)

    if st.button("🚀 开始编译并训练 RNN 网络模型", use_container_width=True, type="primary"):
        if not poem_text.strip():
            st.error("您尚未注入任何文本语料，请补全后重试！")
        else:
            with st.spinner("⏳ 极速扫描字符集并切割时序窗口中..."):
                chars = sorted(list(set(poem_text)))
                vocab_size = len(chars)
                char2int = {c: i for i, c in enumerate(chars)}
                int2char = {i: c for i, c in enumerate(chars)}
                
                # Auto-regressive 数据滑窗构造
                seq_length = 10
                X_data = []
                y_data = []
                for i in range(0, len(poem_text) - seq_length):
                    seq_in = poem_text[i:i + seq_length]
                    seq_out = poem_text[i + 1:i + seq_length + 1]
                    X_data.append([char2int[c] for c in seq_in])
                    y_data.append([char2int[c] for c in seq_out])
                    
                X = torch.tensor(X_data, dtype=torch.long)
                y = torch.tensor(y_data, dtype=torch.long)
            
            st.success(f"数据预处理完毕！全局词表大小 $V={vocab_size}$，共切分得到连续特征序列片段 {len(X_data)} 个。")
            
            # 模型初始化体系
            model = CharRNN(vocab_size, hidden_size)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            st.markdown("#### 📈 实时反向传播 Loss 监控")
            
            progress_bar = st.progress(0)
            chart_placeholder = st.empty()
            loss_history = pd.DataFrame(columns=["Cross Entropy Loss"])
            
            start_time = time.time()
            for epoch in range(1, epochs + 1):
                hidden = model.init_hidden(X.size(0))
                optimizer.zero_grad()
                
                output, hidden = model(X, hidden)
                loss = criterion(output.view(-1, vocab_size), y.view(-1))
                loss.backward()
                optimizer.step()
                
                # Streamlit UI 限频极速流式推送 (避免由于密集渲染导致的卡顿)
                if epoch % max(1, epochs // 50) == 0 or epoch == epochs:
                    progress_bar.progress(epoch / epochs)
                    new_df = pd.DataFrame({"Cross Entropy Loss": [loss.item()]}, index=[epoch])
                    if loss_history.empty:
                        loss_history = new_df
                        chart_placeholder.line_chart(loss_history, height=200)
                    else:
                        chart_placeholder.add_rows(new_df)
                        loss_history = pd.concat([loss_history, new_df])
                        
            st.success(f"🎉 训练完成！网罗特征底座凝固，最终收敛 Loss: `{loss.item():.4f}`，硬件推演耗时: `{time.time()-start_time:.2f} 秒`。")
            
            # 使用 Session 保存权重以抵御重排
            st.session_state.trained_model = model
            st.session_state.char2int = char2int
            st.session_state.int2char = int2char
            st.session_state.is_trained = True
            
    st.markdown("<hr style='margin-top: 10px; margin-bottom: 20px;'>", unsafe_allow_html=True)
    if st.session_state.is_trained:
        st.markdown("#### ✨ 神经网络自回归采样解码 (Auto-regressive Text Generation)")
        seed_char = poem_text.strip().split()[0] if poem_text.strip() else "I"
        
        with st.container():
            seed_col, btn_col = st.columns([3, 1])
            with seed_col:
                seed_input = st.text_input("注入前置引子 (Seed Prompt):", value=seed_char)
            with btn_col:
                st.markdown("<br>", unsafe_allow_html=True)
                gen_btn = st.button("🪄 连环生成 200 个自回归预测字符", use_container_width=True)
            
        if gen_btn:
            if not seed_input:
                st.error("起底前置引子不能为空！")
            else:
                model = st.session_state.trained_model
                model.eval()
                char2int = st.session_state.char2int
                int2char = st.session_state.int2char
                
                generated_text = seed_input
                hidden = model.init_hidden(1)
                
                # 安全性拦截过滤未见字符
                valid_seed = ""
                for c in seed_input:
                    if c in char2int:
                        valid_seed += c
                        
                if not valid_seed:
                    st.error("输入的 Seed 中没有任何字符曾在前面训练的词表内！无法启动寻路算法。")
                else:
                    # RNN 热机态推进：不断吞噬起始序列以重塑隐藏层先验
                    with torch.no_grad():
                        for char in valid_seed[:-1]:
                            x = torch.tensor([[char2int[char]]], dtype=torch.long)
                            _, hidden = model(x, hidden)
                            
                        # 对齐最后一个确切观测到的字符
                        x = torch.tensor([[char2int[valid_seed[-1]]]], dtype=torch.long)
                        
                        # 自回归滚雪球输出
                        for _ in range(200):
                            output, hidden = model(x, hidden)
                            next_index = torch.argmax(output[0, -1]).item()
                            next_char = int2char[next_index]
                            generated_text += next_char
                            
                            x = torch.tensor([[next_index]], dtype=torch.long)
                    st.session_state.m2_gen_text = generated_text
                    
        if st.session_state.get("m2_gen_text"):
            res_txt = st.session_state.m2_gen_text
            st.markdown(f'''
            <div style="padding:22px; background:linear-gradient(90deg, #fefce8 0%, #fef3c7 100%); border-left:6px solid #eab308; border-radius:10px; font-family: 'Courier New', Courier, monospace; font-size: 1.25em; color: #374151; font-weight: bold; box-shadow: 0 4px 10px rgba(0,0,0,0.05); text-align: left; line-height: 1.8;">
                <span style="color:#d97706;">[Seed]</span> {seed_input} <span style="color:#4b5563;">{res_txt[len(seed_input):]}</span>
            </div>
            ''', unsafe_allow_html=True)
    else:
        st.info("模型脑叶空白。请先在上方定制超参并启动训练引擎，点亮这块深度学习区域！")

elif st.session_state.current_page == PAGE_3:
    st.markdown("### 🆚 预训练架构终极对决：Masked LM vs. Causal LM")
    st.markdown("本模块通过真实调用行业开源标杆预训练模型，深度体验 **双向上下文补全 (BERT)** 与 **单向时序自回归流式生成 (GPT-2)** 在基座机理上的本质区别。")
    
    # === 极危防御：超大参数模型防重载并发缓冲层 ===
    @st.cache_resource(show_spinner="⏳ 首次加载千兆级大参数预训练模型中 (后台正在下载 BERT 与 GPT-2 权重矩阵，耗时约1-2分钟请耐心等待)...")
    def load_pipelines():
        try:
            # 引入最经典的遮蔽预言机
            mask_filler = pipeline("fill-mask", model="bert-base-uncased")
            # 引入最经典的短文生成机（禁用旧版生成截断告警）
            text_generator = pipeline("text-generation", model="gpt2")
            return mask_filler, text_generator
        except Exception as e:
            st.error(f"❌ 加载 Hugging Face 远程权重时遭遇严重的网络或内存读写异常。异常日志：{e}")
            return None, None
            
    mask_filler, text_generator = load_pipelines()
    
    if mask_filler and text_generator:
        # 使用并排对称版式渲染强烈的实验对垒氛围
        col_bert, col_gpt2 = st.columns(2, gap="large")
        
        # ----------------- 🎭 BERT 左脑：完形填空 -----------------
        with col_bert:
            st.markdown("<h4 style='color: #15803d;'>🎭 BERT (Masked LM: 掩码预测)</h4>", unsafe_allow_html=True)
            st.info("💡 **上帝视角**：它不受时间向前的单向束缚。由于采用双向 Attention 机制，它能同时阅读并在意 `[MASK]` 词条前后的线索来进行精准推理。多用于事实抽取、分类。")
            
            mask_text = st.text_area("注入时空穿梭探针 (必须留有 `[MASK]` 坑位):", value="The man went to the [MASK] to buy some milk.", height=125)
            
            if st.button("🔍 引爆 BERT 分析器", use_container_width=True):
                if "[MASK]" not in mask_text:
                    st.warning("⚠️ 安全拦截：雷达探测到您的句子中根本没有留下 `[MASK]` 标签占位！BERT 无处降落发力！")
                else:
                    with st.spinner("BERT 正在逆向拆解前后文联结律..."):
                        st.session_state.m3_bert_preds = mask_filler(mask_text)
                        
            if st.session_state.get("m3_bert_preds"):
                st.markdown("<br><b>🏆 预测置信度 Top-5 分布靶场：</b>", unsafe_allow_html=True)
                for i, p in enumerate(st.session_state.m3_bert_preds):
                    token_str = p['token_str']
                    score = p['score']
                    
                    st.write(f"**Top {i+1}: `{token_str}`**")
                    # 使用原生进度条动态映射学术精度，避免枯燥的纯数字堆砌
                    st.progress(score)
                    st.caption(f"上帝视角推演概率 (Probability): **{score:.4f}**")
                            
        # ----------------- ✍️ GPT-2 右脑：一往无前 -----------------
        with col_gpt2:
            st.markdown("<h4 style='color: #0369a1;'>✍️ GPT-2 (Causal LM: 因果生成)</h4>", unsafe_allow_html=True)
            st.info("💡 **宿命论者**：时间长河只能向前走（Causal）。它每走一步都只认识前文长出的词汇，绝对“盲猜”下一个词并自我吞噬闭环。常用于天马行空的编造续写。")
            
            gpt_text = st.text_area("投喂单向引理种子 (Seed Prompt):", value="The man went to the", height=125)
            
            if st.button("🪄 启动 GPT-2 神经元暴走生成", use_container_width=True):
                if not gpt_text.strip():
                    st.error("起底前置引子为空！无法引发连锁反应！")
                else:
                    with st.spinner("GPT-2 单向自回归幻觉脑补编织中..."):
                        # 设置保护参数，防止大模型陷入复读机或无限暴走撑爆内存
                        gen_res = text_generator(
                            gpt_text, 
                            max_new_tokens=20, 
                            pad_token_id=50256, 
                            num_return_sequences=1
                        )
                        st.session_state.m3_gpt2_preds = gen_res[0]['generated_text']
                        
            if st.session_state.get("m3_gpt2_preds"):
                full_text = st.session_state.m3_gpt2_preds
                # 将被模型脑补出来的后缀切离
                new_text_part = full_text[len(gpt_text):]
                
                st.markdown("<br><b>📝 终端全自动续写长卷：</b>", unsafe_allow_html=True)
                
                # 采用卡片化将 “人类施洗的主核” 和 “AI 发癫的续写” 高效视界隔离
                html_gen = f'''
                <div style="padding: 22px; line-height: 1.9; background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); font-size: 1.1em;">
                    <span style="color: #64748b; font-weight: bold; background: #e2e8f0; padding: 4px 6px; border-radius: 6px;">{gpt_text}</span>
                    <span style="color: #0284c7; font-weight: 500; font-family: 'Georgia', serif;">{new_text_part}</span>
                </div>
                '''
                st.markdown(html_gen, unsafe_allow_html=True)

elif st.session_state.current_page == PAGE_4:
    st.markdown("### 🧠 语言模型底层评价度量：困惑度 (Perplexity, PPL)")
    st.markdown("本模块基于 GPT-2 计算给定语句的真实交叉熵损失（Cross-Entropy Loss），并由此推演宏观 PPL。这将为您量化评估模型对不同复杂句子的“惊讶/困惑”程度边界。")
    
    @st.cache_resource(show_spinner="⏳ 加载底层损失评估系统 (GPT-2 Tokenizer & CausalLM)...")
    def load_ppl_model():
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            # 填坑点：GPT-2 极其特殊，默认词表居然不包含 pad_token，强行运算矩阵对齐会报错。因此统一指定用 EOS 代替 pad 以满足张量扩充律。
            tokenizer.pad_token = tokenizer.eos_token
            # 调用带标签（Label）反向反馈接口的专用大模型类
            model = AutoModelForCausalLM.from_pretrained("gpt2")
            model.eval()  # 完全锁定 Dropout 与 Batch Norm
            return tokenizer, model
        except Exception as e:
            st.error(f"❌ 部署系统级量度基座遭遇错误：{e}")
            return None, None
            
    tokenizer, ppl_model = load_ppl_model()
    
    if tokenizer and ppl_model:
        st.markdown("#### 1. 铺陈探测句阵列")
        # 默认三句：日常句、语序错乱句、高密尖端科技句
        default_sentences = "The weather is very beautiful today.\nToday beautiful very is weather the.\nQuantum computing utilizes entanglement and superposition to perform parallel calculations."
        text_lines = st.text_area("输入测试文本池 (一行代表一次独立 PPL 断言)：", value=default_sentences, height=130)
        
        st.markdown("#### 2. 学术标尺投影")
        st.latex(r"PPL(W) = \exp \left( \frac{1}{N} \sum_{i=1}^{N} -\log P(w_i | w_1...w_{i-1}) \right)")
        
        if st.button("📊 重核计算全列困惑度 (Calculate Auto-regressive PPL)", use_container_width=True, type="primary"):
            sentences = [s.strip() for s in text_lines.split("\n") if s.strip()]
            
            if not sentences:
                st.warning("⚠️ 没有待处理标本！")
            else:
                import math
                records = []
                with st.spinner("底层梯度引擎剥离中：正高速提取并对齐全局交叉熵张量块..."):
                    for sent in sentences:
                        # 极度精简提取 input_ids 作为张量供喂食
                        inputs = tokenizer(sent, return_tensors="pt")
                        
                        # 【灵魂防御】：必须严格约束在不求导的空间计算，否则会因为强行拉扯巨量计算图而撑挂云服务器内存
                        with torch.no_grad():
                            # AutoModelForCausalLM 如果收到了 labels 会自动比对生成 CrossEntropy
                            outputs = ppl_model(inputs.input_ids, labels=inputs.input_ids)
                            loss = outputs.loss.item()
                            # 指数化交叉熵暴兵膨胀出直观的倍率 PPL
                            ppl = math.exp(loss)
                            
                        records.append({
                            "测试句子 (Sentence)": sent,
                            "交叉熵损失 (Cross Entropy Loss)": loss,
                            "困惑度 (Perplexity)": ppl
                        })
                
                st.session_state.m4_df = pd.DataFrame(records)
                
        if st.session_state.get("m4_df") is not None:
            df = st.session_state.m4_df
            
            # 剔除对 Matplotlib 依赖的 background_gradient，手写极其轻量级的分段热力场着色盘
            def highlight_ppl(s):
                styles = []
                for v in s:
                    if pd.isna(v):
                        styles.append('')
                    elif v < 300:
                        styles.append('background-color: #dcfce7; color: #166534; font-weight: bold')
                    elif v < 1000:
                        styles.append('background-color: #fef9c3; color: #854d0e; font-weight: bold')
                    else:
                        styles.append('background-color: #fee2e2; color: #991b1b; font-weight: bold')
                return styles

            df_styled = df.style.format({
                "交叉熵损失 (Cross Entropy Loss)": "{:.4f}",
                "困惑度 (Perplexity)": "{:.2f}"
            }).apply(highlight_ppl, subset=["困惑度 (Perplexity)"])
            
            st.markdown("#### 3. 张量微积解算盘")
            st.dataframe(df_styled, use_container_width=True)
            
            # 总结式 AI 分析框
            st.info("💡 **高维可解释性分析简报**：\n\n数据热力场非常清晰地剥离了句法的难度维度！\n- 当遭遇如同 `The weather is very beautiful` 这样的高频习惯发音句，损失极地，PPL 指数呈深绿色健康底色，代表模型对其对答如流，“毫不困惑”。\n- 而针对**强行语法倒装、乱序拼凑的乱码**，或者**包含极其冷僻、突兀堆叠的学术概念的长尾长句**，语言模型在每一步踩碎下一步词汇时都步履维艰、疯狂猜错（巨大的累加对数损失 $Loss$），从而被指数计算法则剧烈放大，直接飙红爆炸出几千乃至上万的惊天 PPL！这也是工业界排查垃圾刷榜文本的照妖镜。")
