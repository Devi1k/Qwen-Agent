import json
import os
import random
import re
from typing import Dict, Optional, Union, List
import faiss
import json5
import numpy as np
import pandas as pd

from qwen_agent.llm.schema import ToolCall, ToolResponse
from qwen_agent.tools.base import BaseTool, register_tool

ROOT_RESOURCE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'agents/resource')


@register_tool('产品推荐')
class Recommend(BaseTool):
    description = '调用理财/产品推荐系统'
    parameters = [{
        'name': 'product_style',
        'type': 'string',
        'description': '产品风格',
        'enums': ['均衡', '偏大盘', '中波动固收+', '偏中短债', '偏成长', '大小盘平衡', '偏中短债', '大小盘平衡 ',
                  '科技主题', '偏小盘', '医药主题', '大小盘平衡']
    }, {
        'name': 'risk_level',
        'type': 'string',
        'description': '风险等级',
        'enums': ['低风险', '中低风险', '中风险', '中高风险', '高风险']
    }, {
        'name': 'investment_sector',
        'type': 'string',
        'description': '投资板块',
        'enums': ['周期', '金融地产', '消费', '科技', '制造', '医药']
    }]

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.df = pd.read_excel(ROOT_RESOURCE + "/基金表.xls")

        self._build_index()

    def call(self, params: Union[str, dict], **kwargs) -> ToolResponse:
        params = self._verify_json_format_args(params)
        product_style, risk_level, investment_sector = params.get('product_style', ""), params.get('risk_level',
                                                                                                   ""), params.get(
            'investment_sector', "")
        recommend_list = set()

        for ps in product_style.split(","):
            if product_style.split(","):
                try:
                    for k in self.style_index[ps]:
                        recommend_list.add(k)
                except KeyError:
                    pass

        for r_l in risk_level.split(","):
            if risk_level.split(","):
                try:
                    for k in self.risk_index[r_l]:
                        recommend_list.add(k)
                except KeyError:
                    pass

        for i_s in investment_sector.split(","):
            if investment_sector.split(","):
                try:
                    for k in self.sector_index[i_s]:
                        recommend_list.add(k)
                except KeyError:
                    pass
        tool_res = ToolCall(self.name, self.description, params, [])
        if kwargs.get("flag", None): recommend_list = random.sample(self.df["基金简称"].tolist(), 5)
        if len(recommend_list) > 5:
            recommend_res = {"推荐内容": random.sample(recommend_list, 5)}
        elif len(recommend_list) != 0:
            recommend_res = list(recommend_list)
        else:
            return ToolResponse(reply="暂无相关产品推荐")
        tool_res.observation = recommend_res
        return ToolResponse("", tool_res)

    def _build_index(self):

        # 将NaN值替换为空字符串
        df = self.df.fillna('')

        # 初始化空字典来存储索引
        style_index = {}
        sector_index = {}
        risk_index = {}

        # 根据产品风格建立索引
        for index, row in df.iterrows():
            styles = row['产品风格'].split(',')
            for style in styles:
                if style not in style_index:
                    style_index[style] = []
                style_index[style].append(row['基金简称'])

        # 根据投资板块建立索引
        for index, row in df.iterrows():
            sectors = row['投资板块'].split(',')
            for sector in sectors:
                if sector not in sector_index:
                    sector_index[sector] = []
                sector_index[sector].append(row['基金简称'])

        # 根据基金风险等级建立索引
        for index, row in df.iterrows():
            risks = row['基金风险等级'].split(',')
            for risk in risks:
                if risk not in risk_index:
                    risk_index[risk] = []
                risk_index[risk].append(row['基金简称'])
        if "" in style_index:
            style_index = style_index.pop("")
        if "" in sector_index:
            del sector_index[""]
        if "" in risk_index:
            risk_index = risk_index.pop("")

        self.style_index = style_index
        self.sector_index = sector_index
        self.risk_index = risk_index
