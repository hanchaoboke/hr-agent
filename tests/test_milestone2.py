import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# 从tools包中导入函数
from agent.rag_pipeline2 import search_hr_policy

if __name__ == '__main__':
    print('-------- RAG Pipeline Test -----------')

    questions = [
        'P5员工去成都出差，一天住宿报销多少钱？',
        '入职半年的新人公司有福利假么？',
        '我想开收入证明，可以自己再系统里弄么？'
    ]

    for i, question in enumerate(questions, 1):
        print(f'\n [测试问询 {i}]: {question}')
        print('-'* 50)

        # 直接调用封装好的 tool，模拟 Agent 图中的操作
        result = search_hr_policy.invoke({'query': question})
        print(result)
        print('-'* 50)

