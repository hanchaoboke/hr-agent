"""
rag 管道
"""
import os
from pathlib import Path
from langchain_core.tools import tool
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# 工程路径配置
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCPATH = PROJECT_ROOT / "data" / 'company_handbook.md'
VECTOR_DB_PATH = PROJECT_ROOT / 'db' / 'chroma_db'

# 1. 初始化 Embedding 模型
print('正在加载 BGE 向量模型。。。。。。')
embeddings = HuggingFaceEmbeddings(
    model_name=os.getenv("EMBEDDING_MODEL"),
    model_kwargs={'device': 'mps'},                 # 模型使用的设备，可以是 cpu、cuda、mps
    encode_kwargs={'normalize_embeddings': True},   # BGE 模型推荐开启归一化
)

def init_vector_store() -> Chroma:
    """初始化向量数据库。如果存在则读取，如果不存在则切分文档并生成"""
    if VECTOR_DB_PATH.exists() and any(VECTOR_DB_PATH.iterdir()):
        # 以经存在向量数据，直接加载
        return Chroma(persist_directory=str(VECTOR_DB_PATH), embedding_function=embeddings)

    print('未检测到本地向量库，开始构建 RAG 索引......')

    if not DOCPATH.exists():
        raise FileNotFoundError(f'找不到知识库文件：{DOCPATH}')

    with open(DOCPATH, 'r', encoding='utf-8') as f:
        markdown_text = f.read()

    headers_to_split_on = [
        ('##', 'Chapter'),
        ('###', 'Section'),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_text)
    # 为了防止某个章节内容依然过长，再叠加一个字符级的滑动窗口切分
    chunk_size = 500
    chunk_overlap = 50
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(md_header_splits)

    print(f'文档切分完毕，共生成 {len(splits)} 个语义块（chunks）。正在存入 Chroma 数据库.......')

    vectorestore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(VECTOR_DB_PATH),
    )

    print(f'向量数据库构建成功，落盘至： {VECTOR_DB_PATH}')
    return vectorestore

# 初始化全局数据库实例
vector_store = init_vector_store()
retriever = vector_store.as_retriever(search_kwargs={'k':5})

# 2. 封装为供 LangGraph 调用的工具
@tool
def search_hr_policy(query:str)->str:
    """
    搜索公司规章制度、差旅报销、假期政策、福利等相关信息的必要工具
    输入参数(query) 必须是从员工问题中提炼出精确检索词
    """
    docs = retriever.invoke(query)
    if not docs:
        return '知识库未检测到相关政策，请提示用户询问HR人工'

    # 组装上下文
    context_parts = []
    for i,  doc in enumerate(docs, 1):
        chapter = doc.metadata.get('Chapter', '未知章节')
        secetion = doc.metadata.get("Section", '未知段落')
        context_parts.append(f'[来源{i}] {chapter} >  {secetion}\n {doc.page_content}')

    merged_context = '\n\n'.join(context_parts)

    return f'【知识库检索结果】\n{merged_context}'