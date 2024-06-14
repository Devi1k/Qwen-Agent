import json
import os
from typing import Dict, Optional, Union, List

import faiss
import numpy as np
import pandas as pd

from qwen_agent.settings import DEFAULT_FAQ_SEARCHERS
from qwen_agent.llm.schema import ToolResponse, Message, SYSTEM
from qwen_agent.tools.base import BaseTool, register_tool, TOOL_REGISTRY
from qwen_agent.utils.str_processing import rm_json_md

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


@register_tool('faq')
class GetFAQRes(BaseTool):
    description = '获取用户问题相关 FAQ'
    parameters = [{
        'name': 'query',
        'type': 'str',
        'description': '用户问题',
        'required': True
    }]

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.faq_searchers = self.cfg.get('faq_searchers', DEFAULT_FAQ_SEARCHERS)

        self.search_objs = [TOOL_REGISTRY[name](cfg) for name in self.faq_searchers]

    def call(self, params: Union[str, dict], **kwargs) -> ToolResponse:
        params = self._verify_json_format_args(params)
        query = params.get('query', "")
        if query == "":
            return ToolResponse(False, "请输入相关内容")
        recall_res_list = []
        try:
            for s_obj in self.search_objs:
                recall_res_list.extend(s_obj.call(query=query))
        except Exception as e:
            print(f"搜索调用失败: {e}")
            return ToolResponse(False, "搜索错误")
        if not recall_res_list:
            return ToolResponse(False, "没有搜索结果")

        # format prompt and call llm
        llm = kwargs.get('llm', None)
        reranker_res = ""
        if llm:
            messages = self._build_prompt(recall_res_list)
            reranker_res = list(llm.chat(messages=messages))[-1][0].content
            if "```" in reranker_res:
                reranker_res = rm_json_md(reranker_res)

        return ToolResponse(False, reranker_res)

    def _build_prompt(self, query: str, recall_res_list: List[Dict]) -> List[Message]:
        AVAILABLE_FAQ = """
        - {'编号': 1, '问题': '什么是最大回撤比例', '答案': '用于衡量投资产品（如基金、股票、投资组合等）风险的重要指标。它表示在特定时间段内，投资产品从最高点到最低点的最大跌幅，通常用百分比表示。最大回撤比例反映了投资产品在最差情况下可能经历的最大损失。', '参考相似问': ['最大回撤是什么']}
        - {'编号': 2, '问题': '开放式基金是什么', '答案': '开放式基金是一种灵活的投资基金类型，其份额数量不是固定的，可以根据投资者的需求进行随时的申购（购买）和赎回（卖出）。', '参考相似问': []}
        - {'编号': 3, '问题': '基金赎回一般要多久到账', '答案': '货币市场基金通常T+1日到账，股票型基金、混合型基金、债券型基金一般为T+3日到账，QDII基金（境外投资基金）通常T+7日或更长时间到账', '参考相似问': ['我赎回基金后多久能到']}
        """
        for i in range(len(recall_res_list)):
            AVAILABLE_FAQ += f"\\{'编号': {i + 4}, '问题': {recall_res_list[i]['query']},'答案': {recall_res_list[i]['content']}\\}\n"
        user_prompt = "## Input :\nuser: " + query + "\n解析结果为:"
        messages = [Message(**{"role": "user", "content": user_prompt})]
        messages.insert(0, Message(SYSTEM, SYSTEM_PROMPT.replace("{available_faqs}", AVAILABLE_FAQ)))
        return messages


@register_tool('faq_embedding')
class GetFAQ(BaseTool):

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.recall_set = pd.read_excel(ROOT_RESOURCE + "/faq.xlsx", sheet_name=0, header=0)
        if os.path.exists(ROOT_RESOURCE + "/faq.index"):
            self.recall_embedding = faiss.read_index(ROOT_RESOURCE + "/faq.index")
        else:
            self.recall_embedding = self._build_index(self.embedding_model, self.all_product)

    def call(self, params: Union[str, dict], **kwargs) -> List[Dict]:
        params = self._verify_json_format_args(params)
        query = params['query']
        query_embedding = self.embedding_model.encode(query, normalize_embeddings=True).reshape(1, -1)
        score, prd_index = self.recall_embedding.search(query_embedding.astype(np.float32), k=5)
        prd_index = prd_index[score > 0.6]
        if len(prd_index) != 0:
            candidate_faq = self.recall_set.iloc[prd_index].to_dict(orient="records")
            return [candidate_faq]
        else:
            return []

    def _build_index(self, embedding_model, df: pd.DataFrame):
        recall_embedding = self.embedding_model.encode(df["query"].tolist(), normalize_embeddings=True)
        dim = recall_embedding.shape[-1]
        # 根据嵌入向量的维度创建Faiss索引，使用内积作为度量标准
        faiss_index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
        # 将嵌入向量转换为float32类型，以满足Faiss的要求
        recall_embedding = recall_embedding.astype(np.float32)
        # 训练Faiss索引，以便其可以接受和处理嵌入向量
        faiss_index.train(recall_embedding)
        # 将所有嵌入向量添加到索引中
        faiss_index.add(recall_embedding)
        # 返回构建好的Faiss索引
        faiss.write_index(faiss_index, ROOT_RESOURCE + "/faq.index")
        return faiss_index
