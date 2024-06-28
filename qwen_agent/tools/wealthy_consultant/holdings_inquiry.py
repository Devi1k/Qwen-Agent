import json
import os
from typing import Dict, Optional, Union

from qwen_agent.llm.schema import ToolResponse, ToolCall
from qwen_agent.tools.base import BaseTool, register_tool

ROOT_RESOURCE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'agents/resource')


@register_tool('持仓查询')
class HoldingsInquiry(BaseTool):
    description = '该工具用于查询用户持仓产品以及对应持仓份额'
    parameters = []

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.holding_path = ROOT_RESOURCE + "/持仓.json"

    # todo: 传入用户id
    def call(self, params: Union[str, dict], **kwargs) -> ToolResponse:
        params = self._verify_json_format_args(params)
        customer_id = params.get('cstno')
        print(customer_id)
        with open(self.holding_path, "r", encoding="utf8") as fp:
            holdings = json.load(fp)
        tool_res = ToolCall(self.name, self.description, params, [holdings])
        return ToolResponse(reply="", tool_call=tool_res)
