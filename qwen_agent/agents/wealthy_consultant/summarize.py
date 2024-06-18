import copy

from typing import Dict, Iterator, List, Optional, Union, Tuple

import json5

from qwen_agent import Agent
from qwen_agent.llm.base import BaseChatModel
from qwen_agent.llm.schema import CONTENT, DEFAULT_SYSTEM_MESSAGE, Message, ROLE, SYSTEM, USER, Session, Turn
from qwen_agent.tools import BaseTool

DEFAULT_NAME = 'Summarizer'
DEFAULT_DESC = '可以根据给出的会话信息，进行摘要总结'

PROMPT_TEMPLATE_ZH = """
## Role :
- 你是一个面向基金财富领域的小助手，请参考以下的一些信息给出专业性回复建议

##  Constrains :
- kb_results为基金图谱的查询结果，包括基金、基金经理等具体信息
- faqs为相关的问答库，请严格参考答案回复相关问题，不要按照自己的知识进行回复
- function_calling_results为调用外部系统的执行结果，请参考return当中的结果

##  Reply :
- 请结合History和Input当中的信息，分析用户的需求进行回复
- 如果用户输入中表达了一些投资失败的一些负面情绪，要进行一定的安抚
- 在进行基金等推荐的时候，给出一些风险提示



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


class Summarizer(Agent):
    """This is an agent for Final Summarize"""

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
        messages = copy.deepcopy(messages)
        if messages[0][ROLE] == SYSTEM:
            messages[0][CONTENT] += PROMPT_TEMPLATE[lang]
        else:
            messages.insert(0, Message(SYSTEM, PROMPT_TEMPLATE[lang]))

        # 构建对话历史字符串，包括最近最近三次用户和assistant的对话记录
        # history_str = ""
        # for turn in kwargs["sessions"].turns[-3:]:
        #     history_str += "user:" + turn.user_input + "\n" + "assistant:" + turn.assistant_output + "\n"

        # 如果最后一条消息是用户消息，则在消息前添加对话历史和解析结果提示
        messages = self._build_prompt(messages=messages, sessions=kwargs.get("sessions"), turn=kwargs.get("turn"))

        # 调用大语言模型处理消息列表，并返回处理结果
        output_stream = self._call_llm(messages=messages)
        return output_stream

    def _build_prompt(self, messages: List[Message], sessions: Session = None, turn: Turn = None) -> List[Message]:

        # 对示例字符串进行预处理，准备插入到模板中的内容
        # example_str = self._example_preprocess()
        history_str = sessions.get_whole_history()
        cur_turn_info = turn.__repr__()
        if messages[-1][ROLE] == USER:
            cur_user_input = messages[-1][CONTENT][0].text
            messages[-1][CONTENT][0].text = history_str + "\n" + cur_turn_info + "\n## Input:" + cur_user_input + "\n解析结果为：\n"
            print(messages[-1][CONTENT][0].text)
        return messages
