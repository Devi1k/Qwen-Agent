import os
from typing import Dict, Optional, Union

import pandas as pd
import requests

from qwen_agent.tools.base import BaseTool, register_tool


@register_tool('get_account_info')
class GetAccountInfo(BaseTool):
    description = '获取用户账户信息'
    parameters = [{
        'name': 'cstno',
        'type': 'int',
        'description': '账户号',
        'required': True
    }, {
        'name': 'cstname',
        'type': 'string',
        'description': '账户名',
        'required': True
    }, {
        'name': 'ctfno',
        'type': 'int',
        'description': '身份证号',
        'required': True
    }]

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)

        cstno, ctfno, cstname = params['cstno'], params["ctfno"], params["cstname"]
        # todo: 调用获取账户信息BP接口
        return f'{cstno}的账户名是{cstname}，身份证号是{ctfno}'
        # response = requests.get(self.url.format(city=self.get_city_adcode(location), key=self.token))
        # data = response.json()
        # if data['status'] == '0':
        #     raise RuntimeError(data)
        # else:
        #     weather = data['lives'][0]['weather']
        #     temperature = data['lives'][0]['temperature']
        #     return f'{location}的天气是{weather}温度是{temperature}度。'
