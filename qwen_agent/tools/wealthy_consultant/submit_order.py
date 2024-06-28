import json
import os
from typing import Dict, Optional, Union

from qwen_agent.llm.schema import ToolResponse, ToolCall
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.utils.str_processing import process_param_list

ROOT_RESOURCE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'agents/resource')


@register_tool('提交订单')
class SubmitOrder(BaseTool):
    description = '该工具用于用户购入产品并更新用户持仓信息'
    parameters = [{
        'name': 'product_name',
        'type': 'string',
        'description': '产品名称',
    }, {
        'name': 'purchase_share',
        'type': 'string',
        'description': '购入份额',
    }]

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.holdings = None
        self.holding_path = ROOT_RESOURCE + "/持仓.json"

    # todo: 传入用户id
    def call(self, params: Union[str, dict], **kwargs) -> ToolResponse:
        params = self._verify_json_format_args(params)
        purchase_share, product_name = params.get('purchase_share', ''), params.get("product_name", '')
        purchase_share = process_param_list(purchase_share)
        product_name = process_param_list(product_name)
        with open(self.holding_path, "r", encoding="utf8") as fp:
            self.holdings = json.load(fp)
        try:
            if len(product_name) > 1:
                for prd_name, share in zip(product_name, purchase_share):
                    self.handle_order(prd_name, share)
            else:
                self.handle_order(product_name[0], purchase_share[0])
        except Exception:
            tool_res = ToolCall(self.name, self.description, params, [{"status": 400, "msg": "交易失败"}])
            return ToolResponse(reply="", tool_call=tool_res)
        tool_res = ToolCall(self.name, self.description, params, [{"status": 200, "msg": "交易成功"}])

        with open(self.holding_path, "w", encoding="utf8") as fp:
            json.dump(self.holdings, fp, ensure_ascii=False, indent=4)
        return ToolResponse(reply="", tool_call=tool_res)

    def handle_order(self, product_name, purchase_share):
        if product_name in self.holdings:
            self.holdings[product_name]['持有份额'] = float(self.holdings[product_name]['持有份额']) + float(
                purchase_share)
        elif product_name != "" and product_name not in self.holdings:
            self.holdings[product_name] = {'持有份额': float(purchase_share)}
