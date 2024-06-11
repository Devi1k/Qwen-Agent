import copy
import json
import traceback
from typing import Dict, Iterator, List, Optional, Union, Tuple

import json5

from qwen_agent import Agent
from qwen_agent.llm.base import BaseChatModel
from qwen_agent.llm.schema import CONTENT, DEFAULT_SYSTEM_MESSAGE, Message, ROLE, SYSTEM
from qwen_agent.tools import BaseTool

DEFAULT_NAME = 'Skill Recognizer'
DEFAULT_DESC = '可以根据问题，识别出用户输入信息中需要使用哪种外部工具'


# todo: dynamic example
PROMPT_TEMPLATE_ZH = """
## Role :
- 你是一个面向基金财富领域的意图分析工具，请判断用户输入信息中是否需要使用一些外部工具

## OutputFormat :
- format: json
- json sample: 
```{
    "thought": "",
    "function_call": []
}```

## Available_Functions :
{function_info}

##  Constrains:
- 必须选择Available_Functions定义中的工具，不要编造其他工具
- parameters包含enums时，解析结果必须是enums当中的选项
- 直接输出json
- 先在thought中给出推理过程，然后在function_call中给出结果

##  Examples:
user: 葛兰管理的基金都有哪些
解析结果为:
{"function_call": []}

user: 我最近怎么亏了这么多
解析结果为:
{
    "function_call": [
        {
            "name": "get_account_info",
            "parameters": {
                "cstno": 62121206326456862,
                "cstname": "张三",
                "ctfno": 11010120912085089124
            }
        }
    ]
}

user: 帮我推荐一只股票
解析结果为:
{
    "function_call": [
        {"name": "call_recommendation","parameters": {"recomm_type": "股票"}}
    ]
}

user: 我持仓的基金最近表现怎样
解析结果为:
{
    "function_call": [
        {
            "name": "get_account_info",
            "parameters": {
                "cstno": 62121206326456862,
                "cstname": "张三",
                "ctfno": 11010120912085089124
            }
        }
    ]
}


"""

PROMPT_TEMPLATE_EN = """
Please fully understand the content of the following reference materials and organize a clear response that meets the user's questions.
# Reference materials:
{ref_doc}

"""

PROMPT_TEMPLATE = {
    'zh': PROMPT_TEMPLATE_ZH,
    'en': PROMPT_TEMPLATE_EN,
}


# noinspection PyMethodMayBeStatic
class SkillRecognizer(Agent):
    """This is an agent for Skill Recognizer"""

    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
                 name: Optional[str] = DEFAULT_NAME,
                 description: Optional[str] = DEFAULT_DESC, ):
        super().__init__(function_list=function_list,
                         llm=llm,
                         system_message=system_message,
                         name=name,
                         description=description)

    def _run(self, messages: List[Message], lang: str = 'zh', **kwargs) -> Iterator[List[Message]]:
        # print(messages)

        self.build_prompt(function_map=kwargs["function_map"])
        messages = copy.deepcopy(messages)
        # system_prompt = PROMPT_TEMPLATE[lang].format(user_query=knowledge)
        if messages[0][ROLE] == SYSTEM:
            messages[0][CONTENT] += PROMPT_TEMPLATE[lang]
        else:
            messages.insert(0, Message(SYSTEM, PROMPT_TEMPLATE[lang]))

        output_stream = self._call_llm(messages=messages)
        return output_stream

    def build_prompt(self, function_map: Dict[str], lang: str = 'zh'):
        # todo: read session history
        function_info = [tool.function for tool_name, tool in function_map.items()]
        function_info_str = ""
        for func in function_info:
            function_info_str += "- " + json5.dumps(func, ensure_ascii=False, indent=4) + "\n"
        PROMPT_TEMPLATE[lang].replace("{function_info}", function_info_str)
