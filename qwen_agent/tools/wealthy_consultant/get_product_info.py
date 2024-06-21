import json
import os
from typing import Dict, Optional, Union, List, Tuple

import faiss
import json5
import numpy as np
import pandas as pd

from qwen_agent.llm.schema import Message, ToolResponse, ToolCall, Session
from qwen_agent.llm.schema import SYSTEM
from qwen_agent.tools.base import BaseTool, register_tool

ROOT_RESOURCE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'agents/resource')

PROMPT_TEMPLATE_ZH = """
## Role :
- 你是一个面向基金财富领域的意图分析工具，请分析用户输入信息中基金、基金经理等信息的实体链接结果

## OutputFormat :
- format: json list
- json list sample: 
```
{
  "是否需要澄清": "是/否",
  "链接结果": [
    {
        "type": "基金/基金经理",
        "name": "XXX"
    }
  ]
}
```

## Fields :
- type: 字段类型，类型为string，必须为["基金", "基金经理"]之一
- value: 字段值，类型为string，最终的实体链接结果，如果是基金的话，链接到具体的基金名称上，不要缩写和简称

##  Constrains:
- 不要抽取Fields定义之外的参数
- "相关的基金信息"为召回模型得到的候选基金列表
- "用户持仓的基金"为用户目前的持仓基金列表
- "历史对话"可能包含一些推荐的基金列表
- 请结合对话上下文、用户的持仓基金、候选的相关基金综合判断用户输入中的基金实体为哪一个
- 仅有多个实体对应无法确定，"是否需要澄清"为"是";

##  Examples:
- 示例1: 
相关的基金信息:
[
  {"基金名称": "富国中证A50ETF联接C", "基金代码": "021211"}
  {"基金名称": "富国鑫旺稳健养老A", "基金代码": "006297"}
  {"基金名称": "富国中证新能源汽车指数", "基金代码": "161028"}
]
用户持仓的基金:
[]
历史对话:
User: 帮我推荐一下基金
Assistant: 以下是一些推荐的基金
{"基金名称": "易方达消费行业股票", "基金代码": "110022"}
{"基金名称": "汇添富消费行业混合", "基金代码": "000083"}
{"基金名称": "嘉实消费精选股票", "基金代码": "005621"}
{"基金名称": "富国中证消费50ETF", "基金代码": "515650"}
User: 富国和嘉实哪个收益好
实体链接结果为:
{
  "是否需要澄清": "否",
  "链接结果": [
      {
          "type": "基金",
          "name": "富国中证消费50ETF"
      },
      {
          "type": "基金",
          "name": "嘉实消费精选股票"  
      }
  ]
}

- 示例2:
相关的基金信息:[
  {"基金名称": "华安聚嘉精选混合", "基金代码": "206018"}
  {"基金名称": "华安聚嘉成长股票", "基金代码": "012345"}
  {"基金名称": "华安聚嘉均衡配置", "基金代码": "678900"}
  {"基金名称": "华安聚嘉主题精选", "基金代码": "123456"}
]
用户持仓的基金:
{"基金名称": "华安聚嘉精选混合A", "基金代码": "515650"}
User: 评价一下华安聚家这个基金
实体链接结果为:
{
  "是否需要澄清": "否",
  "链接结果": [
    {
        "type": "基金",
        "name": "华安聚嘉精选混合A"
    }
  ]
}

- 示例3:
相关的基金信息:[]
用户持仓的基金:[{"基金名称": "华安聚嘉精选混合A", "基金代码": "515650"}]
User: 基金评测
实体链接结果为:
{
  "是否需要澄清": "否",
  "链接结果": []
}

- 示例4:
相关的基金信息:[
  {"基金名称": "华安聚嘉精选混合", "基金代码": "206018"}
  {"基金名称": "华安聚嘉成长股票", "基金代码": "012345"}
  {"基金名称": "华安聚嘉均衡配置", "基金代码": "678900"}
  {"基金名称": "华安聚嘉主题精选", "基金代码": "123456"}
]
用户持仓的基金:

历史对话:
User: 华安聚嘉涨了没
实体链接结果为:
{
  "是否需要澄清": "是",
  "链接结果": []
}


"""

PROMPT_TEMPLATE = {
    'zh': PROMPT_TEMPLATE_ZH
}


def _build_clarify_prompt(user_input: str, candidate_prd: List[Dict], user_product: Dict) -> List[Message]:
    candidate_prd_str = [json.dumps(c, ensure_ascii=False, indent=4) for c in candidate_prd]
    clarify_content = ("相关基金信息：\n" + ",".join(candidate_prd_str) + "\n" +
                       "用户持仓的基金：\n" + json.dumps(user_product, ensure_ascii=False, indent=4) + "\n" +
                       "历史对话：\n" + user_input + "\n" + "实体链指结果：" + "\n")
    clarify_content_str = "## Input:\n" + clarify_content
    messages = [Message(**{"role": "user", "content": clarify_content_str})]
    messages.insert(0, Message(SYSTEM, PROMPT_TEMPLATE['zh']))
    return messages


def _get_history(session: Session) -> str:
    history_str = ""
    if session:
        for turn in session.turns[-3:]:
            history_str += "user:" + turn.user_input + "\n" + "assistant:" + turn.assistant_output + "\n"
    return history_str


@register_tool('产品查询')
class GetProductInfo(BaseTool):
    description = '基金/理财产品信息查询'
    parameters = [{
        'name': 'product_type',
        'type': 'string',
        'description': '要推荐的内容类型，必须为enums当中定义的产品类型',
        'enums': ['基金', '理财']
    }, {
        'name': 'field_type',
        'type': 'string',
        'description': '要推荐的字段类型，必须为enums当中定义的字段类型，没有提到则为空',
        'enums': ['产品评测', '加减仓市场分析']
    }, {
        'name': 'product_name',
        'type': 'string',
        'description': '要查询的基金/理财等产品的名称，抽取字段'
    }, {
        'name': 'product_id',
        'type': 'string',
        'description': '要查询的基金/理财等产品的编码，抽取字段'
    }, {
        'name': 'product_manager',
        'type': 'string',
        'description': '要查询的基金/理财经理的人名，抽取字段'
    }]

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        import os

        from sentence_transformers import SentenceTransformer

        self.model_path = ROOT_RESOURCE + "/acge_text_embedding"
        self.embedding_model = SentenceTransformer(self.model_path)
        self.all_product = pd.read_excel(ROOT_RESOURCE + "/基金表.xls", sheet_name=0, header=0)
        if os.path.exists(ROOT_RESOURCE + "/all_product_embedding.index"):
            self.all_product_embedding = faiss.read_index(ROOT_RESOURCE + "/all_product_embedding.index")
        else:
            self.all_product_embedding = self._build_index(self.all_product)
        self.default_field = ["基金简称", "基金编码", "产品风格", "基金风险等级", "投资板块"]

        # self.recommend_tool = Recommend()

    def _verify_json_format_args(self, params: Union[str, dict]) -> Union[str, dict]:
        """Verify the parameters of the function call"""
        try:
            if isinstance(params, str):
                params_json = json5.loads(params)
            else:
                params_json = params
            for param in self.parameters:
                if param['name'] not in params_json:
                    params_json[param['name']] = ""

            return params_json
        except Exception:
            raise ValueError('Parameters cannot be converted to Json Format!')

    def call(self, params: Union[str, dict], **kwargs) -> ToolResponse:
        # call embedding model for product name
        params = self._verify_json_format_args(params)
        prd_name, prd_id, field_type = params.get('product_name', ""), params.get('product_id', ""), params.get(
            "field_type", "")
        clarify_match, candidate_prd = self._get_candidate_prd(prd_name, prd_id)
        # get user product info from params
        user_product = {}

        # build clarify content
        # user_input = _get_history(kwargs.get("session", None)) + "user:" + kwargs["messages"][-1].content[0].text
        # messages = _build_clarify_prompt(user_input=user_input,
        #                                  candidate_prd=clarify_match, user_product=user_product)
        # # call llm for deciding clarify
        # clarify_llm_res = list(kwargs["llm"].chat(messages=messages))[-1][0].content
        # if "```" in clarify_llm_res:
        #     clarify_llm_res = rm_json_md(clarify_llm_res)
        # try:
        #     clarify_res = json5.loads(clarify_llm_res)
        #     clarify_flag, clarify_content = clarify_res["是否需要澄清"] == "是", clarify_res["链接结果"]
        # except Exception as e:
        #     print(clarify_llm_res)
        #     return ToolResponse(reply=clarify_llm_res)
        # # if need clarify Splicing fixed template
        # if len(clarify_res["链接结果"]) > 1 and clarify_flag:
        #     if candidate_prd:
        #         return ToolResponse("请您确定产品信息: \n" + json.dumps(candidate_prd, ensure_ascii=False, indent=4))
        #     else:
        #         clarify_res = self.recommend_tool.call(params={"product_style": "", "risk_level": "",
        #                                                        "investment_sector": ""}, flag=True)
        #         clarify_res.reply = "未找到相关产品，为您推荐以下产品",
        #         return clarify_res
        #     # else return product info field at least five fields
        # else:
        # todo: 根据类型（基金/理财） + 名称区分
        # try:
        #     target_name_list = [cc["name"] for cc in clarify_content]
        # except KeyError:
        #     print(clarify_content)
        #     target_name_list = []
        if isinstance(field_type, str):
            field_type = field_type.split(",")
        desire_fields = [ft if ft != "" and ft in self.parameters[1]["enums"] else "" for ft in field_type]
        query_field_list = self.default_field.copy()
        if desire_fields != "":
            query_field_list.extend(desire_fields)

        # 根据查询字段列表和目标基金名称列表，筛选出匹配的基金数据
        matching_funds = [
            {field: candi[field] for field in query_field_list if field in candi}
            for candi in candidate_prd
        ]

        # matching_funds = [candi for candi in candidate_prd if candi["基金简称"] in target_name_list]

        return ToolResponse(reply="",
                            tool_call=ToolCall(action=self.name, description=self.description, action_input=params,
                                               observation=matching_funds))

    def _build_index(self, df: pd.DataFrame) -> faiss.IndexFlat:
        """
        构建Faiss索引用于基金名称嵌入的相似性搜索。

        使用SentenceTransformer模型将基金名称转换为嵌入向量，并在Faiss中构建索引以支持高效的相似性搜索。

        参数:
        model: SentenceTransformer模型，用于生成基金名称的嵌入向量。
        df: Pandas DataFrame，包含基金名称的DataFrame。

        返回:
        faiss.Index: 已经训练和填充了基金名称嵌入向量的Faiss索引。
        """
        # 将基金名称转换为嵌入向量

        all_prd_name_embedding = self.embedding_model.encode(df['基金简称'].tolist(), normalize_embeddings=True)
        # 获取嵌入向量的维度
        dim = all_prd_name_embedding.shape[-1]
        # 根据嵌入向量的维度创建Faiss索引，使用内积作为度量标准
        faiss_index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
        # 将嵌入向量转换为float32类型，以满足Faiss的要求
        all_prd_name_embedding = all_prd_name_embedding.astype(np.float32)
        # 训练Faiss索引，以便其可以接受和处理嵌入向量
        faiss_index.train(all_prd_name_embedding)
        # 将所有嵌入向量添加到索引中
        faiss_index.add(all_prd_name_embedding)
        # 返回构建好的Faiss索引
        faiss.write_index(faiss_index, ROOT_RESOURCE + "/all_product_embedding.index")
        return faiss_index

    def _get_candidate_prd(self, prd_name: str, prd_id: str) -> Tuple[List[Dict], List[Dict]]:
        candidate_prd = []
        if isinstance(prd_name, list) or isinstance(prd_id, list):
            prd_name_list, prd_id_list = prd_name, prd_id
        else:
            prd_name_list, prd_id_list = prd_name.split(","), prd_id.split(",")

        if len(prd_id_list) < len(prd_name_list):
            for i in range(len(prd_id_list), len(prd_name_list)): prd_id_list.append("")
        else:
            for i in range(len(prd_name_list), len(prd_id_list)): prd_name_list.append("")
        for prd_id, prd_name in zip(prd_id_list, prd_name_list):
            prd_index_res = -1
            if prd_id in self.all_product['基金编码'].tolist():
                prd_index_res = self.all_product['基金编码'].tolist().index(prd_id)
                candidate_prd.append(self.all_product.iloc[prd_index_res].to_dict(orient="records")[0])
                continue
            # model encode product name and calculate cosine similarity with all product name for top 10
            elif prd_name != "":
                prd_name_embedding = self.embedding_model.encode(prd_name, normalize_embeddings=True).reshape(1, -1)
                score, prd_index = self.all_product_embedding.search(prd_name_embedding.astype(np.float32), k=5)
                prd_index_res = prd_index[score > 0.6]
                if len(prd_index_res) != 0:
                    candidate_prd.append(self.all_product.iloc[prd_index_res].to_dict(orient="records")[0])

        desire_fields = ["基金简称", "基金编码"]
        matching_funds = [
            {field: candi[field] for field in desire_fields if field in candi}
            for candi in candidate_prd]
        return matching_funds, candidate_prd
