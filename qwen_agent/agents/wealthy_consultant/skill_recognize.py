import copy
import json
import os
from typing import Dict, Iterator, List, Optional, Union

from qwen_agent import Agent
from qwen_agent.llm.base import BaseChatModel
from qwen_agent.llm.schema import CONTENT, DEFAULT_SYSTEM_MESSAGE, Message, SYSTEM, Session, USER
from qwen_agent.log import logger
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
- 必须在Available_Functions定义的范围内选择工具，不要编造其他工具
- 必须在工具的parameters范围内抽取信息，不要凭空捏造
- parameters包含enums时，解析结果只能从enums当中选择
- 严格按照 OutputFormat 格式输出
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

        skill_message = []
        # 根据提供的函数映射表构建提示信息
        self._build_prompt(function_map=kwargs["function_map"])
        messages = copy.deepcopy(messages)

        skill_message.insert(0, Message(SYSTEM, PROMPT_TEMPLATE[lang] + "\n\n"))
        # 构建对话历史字符串，包括最近最近三次用户和assistant的对话记录
        history_str = self._get_history(kwargs["sessions"])
        # history_str = ""
        # for turn in kwargs["sessions"].turns[-3:]:
        #     history_str += "user:" + turn.user_input + "\n" + "assistant:" + turn.assistant_output + "\n"
        cur_user_input = messages[-1][CONTENT][0].text
        skill_message.append(Message(USER, history_str + "## Input:\nuser:" + cur_user_input + "\n解析结果为:\n"))
        logger.info("*" * 10)
        logger.info(skill_message)
        # 调用大语言模型处理消息列表，并返回处理结果
        output_stream = self._call_llm(messages=skill_message)
        return output_stream

    def _get_history(self, sessions: Session) -> str:
        history_str = "## History:\n"
        for turn in sessions.turns[-3:]:
            tool_res = "tool result:" + "\n"
            for t_r in turn.tool_res:
                tool_res += t_r.tool_call.__str__() if t_r.tool_call is not None else t_r.reply + "\n"
            history_str += ("user:\n" + turn.user_input + "\n" + tool_res + "\n"
                            "assistant:\n" + turn.assistant_output + "\n")
        return history_str

    def _build_prompt(self, function_map: Dict[str, BaseTool], lang: str = 'zh', sessions: Session = None):

        # 对示例字符串进行预处理，准备插入到模板中的内容
        example_str = self._example_preprocess()

        # 构建函数信息列表，通过遍历function_map中的项，提取每个工具的函数
        function_info = [tool.function for tool_name, tool in function_map.items()]

        # 初始化函数信息字符串
        function_info_str = ""
        # 遍历函数信息列表，将每个函数信息格式化为JSON字符串，并追加到函数信息字符串中
        for func in function_info:
            function_info_str += "- " + json.dumps(func, ensure_ascii=False, indent=4) + "\n"

        # 更新语言模板，将函数信息和示例字符串插入到模板中相应的位置
        PROMPT_TEMPLATE[lang] = PROMPT_TEMPLATE[lang].replace("{function_info}", function_info_str).replace("{example}",
                                                                                                            example_str)

    def _example_preprocess(self) -> str:
        with open(os.path.join(ROOT_RESOURCE, 'skill_example.json'), 'r', encoding='utf-8') as f:
            example = json.load(f)
        example_str = ""
        count = 1
        for func_name, examples in example.items():
            for _ex in examples:
                example_str += f"### Example {count}\n" + "user: " + _ex["input"] + "\n" + "解析结果为:\n" + json.dumps(
                    {"thought": "xxxx", "function_call": _ex["output"]},
                    ensure_ascii=False, indent=4) + "\n"
                count += 1
        return example_str
