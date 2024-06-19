import json
import time
from typing import Dict, Iterator, List, Optional, Union, Tuple
import os
import json5

from qwen_agent import Agent
from qwen_agent.llm.base import BaseChatModel
from qwen_agent.llm.schema import DEFAULT_SYSTEM_MESSAGE, Message, Session, Turn, SYSTEM, ASSISTANT

from qwen_agent.tools import BaseTool
from qwen_agent.utils.str_processing import rm_json_md

DEFAULT_NAME = 'FAQ'
DEFAULT_DESC = '检索相关 FAQ 问题'
ROOT_RESOURCE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resource')

SYSTEM_PROMPT = """
## Role :
- 你是一个面向基金财富领域的意图分析工具，请判断用户输入信息中有哪些可以使用已有的FAQ知识库回答

## OutputFormat :
- format: json
- json sample: 
```{
    "faqs": []
}```

## Available_Faqs :
{available_faqs}


##  Constrains:
- 必须在Available_Faqs中进行选择
- 如果用户问题和Available_Faqs都不相关，输出结果为空
- 直接输出json，不要给出解释过程

##  Examples:
user: 葛兰管理的基金都有哪些
解析结果为:
{
    "faqs": []
}

user: 开放基金赎回几天能到
解析结果为:
{
    "faqs": [3]
}

user: 基金的开放式是指什么，赎回几天能到
解析结果为:
{
    "faqs": [2,3]
}

"""



class FAQAgent(Agent):

    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
                 name: Optional[str] = DEFAULT_NAME,
                 description: Optional[str] = DEFAULT_DESC):
        super().__init__(function_list=function_list,
                         llm=llm,
                         system_message=system_message,
                         name=name,
                         description=description)
        self.session = Session(turns=[])

        self.search_objs = [self.function_map[name] for name in function_list]
        self.available_faqs = """
{'编号': 1, '问题': '什么是最大回撤比例', '答案': '用于衡量投资产品（如基金、股票、投资组合等）风险的重要指标。它表示在特定时间段内，投资产品从最高点到最低点的最大跌幅，通常用百分比表示。最大回撤比例反映了投资产品在最差情况下可能经历的最大损失。', '参考相似问': ['最大回撤是什么']}
{'编号': 2, '问题': '开放式基金是什么', '答案': '开放式基金是一种灵活的投资基金类型，其份额数量不是固定的，可以根据投资者的需求进行随时的申购（购买）和赎回（卖出）。', '参考相似问': []}
{'编号': 3, '问题': '基金赎回一般要多久到账', '答案': '货币市场基金通常T+1日到账，股票型基金、混合型基金、债券型基金一般为T+3日到账，QDII基金（境外投资基金）通常T+7日或更长时间到账', '参考相似问': ['我赎回基金后多久能到']}
"""
    def _run(self, messages: List[Message], lang: str = 'zh', **kwargs) -> Iterator[List[Message]]:

        res_messages = Message(role=ASSISTANT, content="")
        if messages[-1].content[0].text.strip() == '':
            raise ValueError("请输入有关信息")
        query = messages[-1].content[0].text.strip()
        recall_res_list = []
        embedding_start = time.time()
        try:
            for s_obj in self.search_objs:
                recall_res_list.extend(s_obj.call({"query": query}))
        except Exception as e:
            raise ValueError(f"搜索调用失败: {e}")
        print(f"embedding time: {time.time() - embedding_start}")
        # format prompt and call llm

        messages = self._build_prompt(query, recall_res_list)
        llm_res = self._call_llm(messages=messages)
        faq_res = list(llm_res)[-1][0].content
        if "```" in faq_res:
            faq_res = rm_json_md(faq_res)
        faq_json_res = {}
        try:
            faq_json_res = json5.loads(faq_res)
        except Exception as e:
            faq_json_res["faq"] = ""
            raise ValueError(f"解析json失败: {faq_res}")
        candidate_faq_dict = {index: a for index, a in enumerate(self.candidate_faq_str.strip().split("\n"))}
        try:
            res_messages.content = json5.dumps([candidate_faq_dict[index - 1] for index in faq_json_res["faqs"]],
                                               ensure_ascii=False)
        except KeyError:
            pass
        yield [res_messages]

    def _build_prompt(self, query: str, recall_res_list: List[Dict]) -> List[Message]:
        faq_str = ""
        for i in range(len(recall_res_list)):
            faq_str += "{" + f"'编号': {i + 4}, '问题': {recall_res_list[i]['标准问']},'答案': {recall_res_list[i]['答案']},'参考相似问': {recall_res_list[i]['相似问']}" + "}\n"
        self.candidate_faq_str = self.available_faqs + faq_str
        user_prompt = "## Input :\nuser: " + query + "\n解析结果为:"
        messages = [Message(**{"role": "user", "content": user_prompt})]
        messages.insert(0, Message(SYSTEM, SYSTEM_PROMPT.replace("{available_faqs}", self.candidate_faq_str)))
        return messages
