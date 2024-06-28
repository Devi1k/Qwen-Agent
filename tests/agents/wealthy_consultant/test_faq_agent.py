from pprint import pprint

import json5
import pytest

from qwen_agent.agents import FAQAgent
from qwen_agent.llm.schema import Session, Message, USER, ContentItem

test_query = [
    '华宝新兴最近能加仓吗',
    '安鑫稳评测一下这个产品',
    '安享惠金的投资风格是什么',
    '推荐几个低风险产品',
    '民生加银有什么加仓建议吗',
    '请帮我比较一下这两个产品',
    '对比一下中银和华安这两款产品',
    '对比一下中银和汇添富这两款产品',
    '查询安鑫稳和中银这两个产品',
    '查下安鑫稳，再给我推几个低风险产品',
    '理财撤单',
    '查历史业绩',
    '解释 CPI',
    '我现在买了华宝，是否是最佳时机'
]

llm_cfg = {'model': 'qwen-max',
           'api_key': 'sk-22a3f18de8c840d79d3d16f821c9a160',
           'model_server': 'dashscope',
           'generate_cfg':
               {'temperature': 0.1, 'top_p': 0.7, 'max_tokens': 1024}
           }
agent = FAQAgent(llm=llm_cfg, function_list=["faq_embedding"])


@pytest.mark.parametrize('text', test_query)
def test_faq_agent(text):
    # messages = [Message({'role': 'user', 'content': [{'text': 'text'}]})]
    messages = [Message(role=USER, content=[ContentItem(text=text)])]
    # messages = [{'role': 'user', 'content': [{'text': text}]}]

    faq_res = list(agent.run(messages=messages, sessions=Session(turns=[]), lang="zh"))[-1][0].content

    try:
        faq_res_json = json5.loads(faq_res)
    except KeyError:
        faq_res_json = faq_res

    if faq_res:
        for _, f_r in faq_res_json.items():
            if f_r["score"] > 0.6:
                pprint(text)
                pprint(
                    {"标准问": f_r["标准问"], "答案": f_r["答案"],
                     "相似性": round(f_r["score"], 2) * 100})
