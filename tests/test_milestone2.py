import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# 从 agent 包中导入 RAG 检索工具
from agent.rag_pipeline2 import search_hr_policy


# =====================================================================
# pytest 用例：验证高精度 RAG 检索链路
#
# 注意：RAG 结果是非确定性的（底层 LLM 做查询扩写 + HyDE，temperature=0.7），
# 所以不能断言具体文字，而是断言“确实检索到了有效结果”：
#   - 返回是非空字符串
#   - 带有 【来源 N】 溯源标记（说明真的召回并重排了文档）
#   - 没有落到“未检测到相关政策”的兜底分支
#
# 运行方式：pytest tests/test_milestone2.py -v
# 提示：每个问题会走一次完整 RAG 链路（扩写→检索→重排），单条约 15 秒，请耐心等待。
# =====================================================================
QUESTIONS = [
    'P5员工去成都出差，一天住宿报销多少钱？',
    '入职半年的新人公司有福利假么？',
    '我想开收入证明，可以自己再系统里弄么？',
]
# @pytest.mark.parametrize——这是 pytest 的写法,把 3 个问题作为参数,一个测试函数自动展开成 3 个独立用例。
# 好处是某个问题挂了,报告会精确告诉你是哪一条挂的,而不是整个函数一起红。

@pytest.mark.parametrize('question', QUESTIONS)
def test_search_hr_policy_returns_sources(question):
    """每个问题都应召回带溯源标记的知识库内容，而非兜底提示"""
    result = search_hr_policy.invoke({'query': question})

    assert isinstance(result, str)
    assert result.strip(), '检索结果不应为空'
    assert '【来源' in result, f'未召回任何知识库来源，实际返回：{result}'
    assert '未检测到相关政策' not in result, '不应落到“未检索到”的兜底分支'


# =====================================================================
# 保留直接运行入口：python tests/test_milestone2.py 也能看到完整检索输出
# =====================================================================
if __name__ == '__main__':
    print('-------- RAG Pipeline Test -----------')

    for i, question in enumerate(QUESTIONS, 1):
        print(f'\n [测试问询 {i}]: {question}')
        print('-' * 50)

        # 直接调用封装好的 tool，模拟 Agent 图中的操作
        result = search_hr_policy.invoke({'query': question})
        print(result)
        print('-' * 50)
