import copy
import json
import os
import traceback
from typing import Dict, Iterator, List, Optional, Union, Tuple

import json5

from qwen_agent import Agent
from qwen_agent.llm.base import BaseChatModel
from qwen_agent.llm.schema import CONTENT, DEFAULT_SYSTEM_MESSAGE, Message, ROLE, SYSTEM, USER, Session
from qwen_agent.tools import BaseTool

DEFAULT_NAME = 'Skill Recognizer'
DEFAULT_DESC = '可以根据问题，识别出用户输入信息中需要使用哪种外部工具'

PROMPT_TEMPLATE_ZH = """
## Role :
- 你是一个面向基金财富领域的意图分析工具，请判断用户输入信息中是否需要使用一些外部工具

## OutputFormat :
- format: json
- json sample: 
```json
{
    "thought": "",
    "function_call": []
}
```

## Available_Functions :
{function_info}

##  Constrains:
- 必须选择Available_Functions定义中的工具，不要编造其他工具
- parameters包含enums时，解析结果必须是enums当中的选项
- 直接输出json
- 先在thought中给出推理过程，然后在function_call中给出结果

##  Examples:
{example}


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

ROOT_RESOURCE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resource')


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

        # 根据提供的函数映射表构建提示信息
        self._build_prompt(function_map=kwargs["function_map"])
        messages = copy.deepcopy(messages)
        if messages[0][ROLE] == SYSTEM:
            messages[0][CONTENT] += PROMPT_TEMPLATE[lang]
        else:
            messages.insert(0, Message(SYSTEM, PROMPT_TEMPLATE[lang]))

        # 构建对话历史字符串，包括最近最近三次用户和assistant的对话记录
        history_str = ""
        for turn in kwargs["sessions"].turns[-3:]:
            history_str += "user:" + turn.user_input + "\n" + "assistant:" + turn.assistant_output + "\n"

        # 如果最后一条消息是用户消息，则在消息前添加对话历史和解析结果提示
        if messages[-1][ROLE] == USER:
            cur_user_input = messages[-1][CONTENT][0].text
            messages[-1][CONTENT][0].text = history_str + "user:" + cur_user_input + "\n解析结果为：\n"

        # 调用大语言模型处理消息列表，并返回处理结果
        output_stream = self._call_llm(messages=messages)
        return output_stream

    def _build_prompt(self, function_map: Dict[str, BaseTool], lang: str = 'zh', sessions: Session = None):

        # 对示例字符串进行预处理，准备插入到模板中的内容
        example_str = self._example_preprocess()

        # 构建函数信息列表，通过遍历function_map中的项，提取每个工具的函数
        function_info = [tool.function for tool_name, tool in function_map.items()]

        # 初始化函数信息字符串
        function_info_str = ""
        # 遍历函数信息列表，将每个函数信息格式化为JSON字符串，并追加到函数信息字符串中
        for func in function_info:
            function_info_str += "- " + json5.dumps(func, ensure_ascii=False, indent=4) + "\n"

        # 更新语言模板，将函数信息和示例字符串插入到模板中相应的位置
        PROMPT_TEMPLATE[lang] = PROMPT_TEMPLATE[lang].replace("{function_info}", function_info_str).replace("{example}",
                                                                                                            example_str)

    def _example_preprocess(self) -> str:
        with open(os.path.join(ROOT_RESOURCE, 'skill_example.json'), 'r', encoding='utf-8') as f:
            example = json.load(f)
        example_str = ""
        for user_input, call_fun in example.items():
            example_str += "user: " + user_input + "\n" + "解析结果为: " + json.dumps({"function_call": call_fun},
                                                                                      ensure_ascii=False) + "\n"
        return example_str
