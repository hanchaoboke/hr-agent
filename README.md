# Enterprise HR Smart Assistant (多智能体 HR 专家系统)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-Integration-green)](https://python.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-MultiAgent-orange)](https://python.langchain.com/docs/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)

基于 **LangGraph** 和 **高精度 RAG** 构建的企业级多智能体 HR 服务中台。本项目采用 `Actor-Critic` (执行者-审计者) 架构，通过大模型的自我反思（Self-Reflection）机制，彻底解决了传统企业知识库问答中“大模型捏造事实”的痛点，并提供了 Web 交互与 RESTful API 两种调用方式。

## ✨ 核心亮点 (Core Features)

- 🧠 **Actor-Critic 反思架构**：引入独立的合规审计 Agent，实时拦截并打回虚假生成的数字与政策，保证企业规章解读 100% 忠于原文。
- 🔍 **究极 RAG 流水线**：
  - 基于 Markdown 语义层级的文档切分。
  - **同语义多维扩写 (Multi-Query) + HyDE** 技术提高长尾问题召回率。
  - **混合检索** (BM25 + Chroma 向量检索)。
  - 引入 **BGE-Reranker** 交叉编码器进行 Top-3 精准重排。
- 💾 **多轮会话与记忆管理**：内置 LangGraph `MemorySaver`，支持上下文指代消解；结合后台守护线程，实现会话超时自动总结与记忆压缩。
- 🛠️ **智能工具路由**：自动结合员工身份上下文（职级、属地）动态匹配报销标准和假期余额。
- 🚀 **微服务就绪**：提供基于 Streamlit 的可视化演示前端，以及基于 FastAPI 的生产级高并发 API 接口。

---

## 🏗️ 系统架构图 (Architecture)

![这里写图片描述](/docs/architecture.png)

---

## 💻 技术栈 (Tech Stack)

* **应用编排:** LangChain, LangGraph
* **大模型驱动:** OpenAI API / DeepSeek (可无缝切换)
* **向量模型 & 重排:** BAAI/bge-base-zh-v1.5, BAAI/bge-reranker-base
* **向量数据库:** ChromaDB
* **后端服务:** FastAPI, Uvicorn, SQLite3 (Mock DB)
* **前端展示:** Streamlit

---

## 📂 项目结构 (Project Structure)

```text
hr_agent_project/
├── data/                      # 静态资源与知识库文件
│   └── company_handbook.md    
├── database/                  # 数据库配置与初始化
│   └── mock_db.py             
├── tools/                     # Agent 外部工具封装
│   └── hr_tools.py            
├── agent/                     # 核心路由与认知大脑
│   ├── rag_pipeline.py        # 混合检索与重排流水线
│   └── graph_builder.py       # LangGraph 状态机编排
├── tests/                     # 单元测试与评估脚本
├── app.py                     # Streamlit 交互式前端界面
├── api.py                     # FastAPI 后端服务接口
├── .env                       # 环境变量配置 (需自行创建)
└── requirements.txt           # 依赖清单

```

---

## 🚀 快速开始 (Quick Start)

### 1. 环境准备

克隆本项目并安装依赖：

```bash
git clone [https://github.com/your-username/hr-agent-project.git](https://github.com/your-username/hr-agent-project.git)
cd hr-agent-project
pip install -r requirements.txt

```

### 2. 配置环境变量

在项目根目录创建 `.env` 文件，填入你的大模型 API 密钥：

```env
DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
# DEEPSEEK_API_KEY="[https://api.deepseek.com/v1](https://api.deepseek.com/v1)" # 若使用代理或其他兼容接口请配置此项

```

### 3. 初始化数据库与向量库

首次运行前，需生成 Mock 数据库并构建 Chroma 向量索引（会自动下载 BGE 模型权重）：

```bash
python database/mock_db.py
python tests/test_milestone2.py 

```

### 4. 启动可视化前端 (演示模式)

```bash
streamlit run app.py

```

浏览器将自动打开 `http://localhost:8501`，即可与 HR 助理进行对话。

### 5. 启动 API 服务 (生产模式)

```bash
python api.py

```

访问 `http://localhost:8000/docs` 查看 Swagger 交互式接口文档。

---

## 🤝 贡献与参与

本项目为教学演示与工业落地探索的实践项目，欢迎提交 Pull Request 或 Issue 共同完善！

---
