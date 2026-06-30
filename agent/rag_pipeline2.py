import os
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

# 工程路径配置
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCPATH = PROJECT_ROOT / "data" / 'company_handbook.md'
VECTOR_DB_PATH = PROJECT_ROOT / 'db' / 'chroma_db'

DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
RERANKER_MODEL = os.getenv("RERANK_MODEL")

print('正在加载 BGE 向量模型。。。。。。')
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'mps'},                 # 模型使用的设备，可以是 cpu、cuda、mps
    encode_kwargs={'normalize_embeddings': True},   # BGE 模型推荐开启归一化
)

print('初始化重排模型。。。。。。')
reranker = CrossEncoder(
    model_name=RERANKER_MODEL,
    max_length=512,
    device='mps',
)

llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.7                 # 用于Query改写、HyDE生成，温度设为0。7 激发一定的创造力
)

# ======== 2. 构建多路召回 Retriever ======
def build_ensemble_retriever():
    """构建 BM25 + Vector 的混合检索器"""
    with open(DOCPATH, 'r', encoding='utf-8') as f:
        markdown_text = f.read()

    # 第一层：基于 Markdown 语义层级切分
    headers_to_split_on = [
        ('##', 'Chapter'),
        ('###', 'Section'),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_text)

    # 第二层：字符集滑动窗口切分（兜底）
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(md_header_splits)

    print(f'文档切分完毕，共生成 {len(splits)} 个语义块（chunks）。正在存入 Chroma 数据库.......')

    # 路线A：BM25 全文关键字检索
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 5                                    # BM25 召回 5 篇

    # 路线B：向量语义检索（Chroma）
    if VECTOR_DB_PATH.exists() and any(VECTOR_DB_PATH.iterdir()):
        print('[知识库构建] 检测到本地持久化向量库，正在加载。。。。。')
        vectorstore = Chroma(persist_directory=str(VECTOR_DB_PATH), embedding_function=embeddings)
    else:
        print('[知识库构建] 本地无缓存，正在生成。。。。。')
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=str(VECTOR_DB_PATH),
        )

    vector_retriever = vectorstore.as_retriever(search_kwargs={'k':5})

    # 混合 使用 EnsembleRetriever ，采用 RRF 倒排秩融合算法
    ensumble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6]      # 权重可调：偏向关键字还是偏向语义，40%依赖关键字，60%依赖语义泛化
    )
    return ensumble_retriever


print('正在构建混合索引。。。。')
retriever = build_ensemble_retriever()

# 3. 智能扩写与 HyDE
class QueryExpansion(BaseModel):
    expanded_questions: list[str] = Field(description='从不同维度扩写3个相关检索词或短语')
    hypothetical_document:str = Field(description='针对该问题的一段假设性、看似专业的官方制度回答片段（允许伪造数字）')

expansion_parser = JsonOutputParser(pydantic_object=QueryExpansion)

def expand_and_hyde(original_query:str) -> list[str]:
    """利用 LLM 生成多维度扩写问题与 HyDE 假设性长文"""
    prompt = ChatPromptTemplate.from_template(
        "你是一个专业的企业HR专家。为了提高知识库的检索命中率，请协助处理用户的原始提问。\n"
        "任务一（多维扩写）：站在不同的视角（如政策名词、审批流程、系统操作）扩写3个相关的查询短语。\n"
        "任务二（HyDE假设）：用官方、严谨的 HR 规章制度口吻，伪造一段回答该问题的文本。不管事实是否正确，重点是模仿‘员工手册’的专业行文风格。\n\n"
        "用户原始提问：{query}\n\n"
        "{format_instructions}"
    )

    chain = prompt | llm | expansion_parser
    try:
        result = chain.invoke({
            'query': original_query,
            'format_instructions':expansion_parser.get_format_instructions()
        })
        print(f'\n原始问题：{original_query}')
        print(f'    ->衍生查询：{result['expanded_questions']}')
        print(f'    ->HyDE伪文：{result["hypothetical_document"]}')

        # 汇总：原问题 + 3 个衍生问题 + 1 个假设文档
        return [original_query] + result['expanded_questions'] + [result["hypothetical_document"]]
    except Exception as e:
        print(f'LLM 调用失败，降级使用基础索引。原因： {e}')
        return [original_query]

# 4. LangGraph 知识库检索节点工具（Tool）
@tool
def search_hr_policy(query:str)->str:
    """
    搜索公司规章制度、差旅报销、假期政策、福利等相关信息的必要工具
    输入参数(query) 必须用户的原始问题
    """
    # 步骤1:获得智能扩写与HyDE的查询矩阵
    search_queries = expand_and_hyde(query)

    # 步骤2:多路并发索引（BM25 + Vector混合）
    all_candidate_docs = []
    for q in search_queries:
        docs = retriever.invoke(q)
        all_candidate_docs.extend(docs)

    # 步骤3:文档去重
    unique_docs = {doc.page_content:doc for doc in all_candidate_docs}.values()
    unique_docs = list(unique_docs)

    if not unique_docs:
        return '[知识库系统]未检测到相关政策，请用户转人工HR'

    # 步骤4:Cross-Encoder 重排,必须用用户的原始问题和召回的文档计算相关性
    sentence_pairs =[[query, doc.page_content] for doc in unique_docs]
    scores = reranker.predict(sentence_pairs)

    scores_docs = list(zip(unique_docs, scores))
    scores_docs.sort(key=lambda x: x[1], reverse=True)  # 按模型打分从高到低排序

    # 步骤5:截取 Top-3 并组装返回文本
    top_3_docs = [doc for doc, _ in scores_docs[:3]]

    context_parts = []
    for i, doc in enumerate(top_3_docs, 1):
        chapter = doc.metadata.get('Chapter', "未知章节")
        section = doc.metadata.get('Section', '未知段落')
        context_parts.append(f'【来源 {i}】{chapter} > {section}\n{doc.page_content}')

    return '【知识库高精度检验结果】\n' + '\n\n'.join(context_parts)


