import copy
import json
from typing import Dict, Iterator, List, Optional, Union

from qwen_agent import Agent
from qwen_agent.llm.base import BaseChatModel
from qwen_agent.llm.schema import CONTENT, DEFAULT_SYSTEM_MESSAGE, Message, SYSTEM, USER, Session, Turn
from qwen_agent.log import logger
from qwen_agent.tools import BaseTool

DEFAULT_NAME = 'Summarizer'
DEFAULT_DESC = '可以根据给出的会话信息，进行摘要总结'

PROMPT_TEMPLATE_ZH = """
## Role :
- 你是中信银行的一个面向基金财富领域的助手小信。请参考以下的一些信息给出专业性回复建议


##  Constrains :
- faqs 为相关的问答库，请参考知识库的答案回复相关问题
- 请参考 observation 当中的结果作为调用外部系统的执行结果


##  Content requirements:
- 请结合 History 和 Input 当中的信息，分析用户的需求进行回复。如果 History 中提到了多个相似的产品名称，请向用户确认, 回复"您提到的xx可能存在多个产品，请选择: 1.xxxx 2.xxxx 3.xxxx ..."
- 如果用户输入中表达了一些投资失败的一些负面情绪，要进行一定的安抚
- 提到具体产品名称的时候，请加上产品代码
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


def _build_prompt(messages: List[Message], sessions: Session = None, turn: Turn = None) -> Message:
    # 对示例字符串进行预处理，准备插入到模板中的内容
    # example_str = self._example_preprocess()
    history_str = sessions.get_whole_history()
    cur_turn_info = "工具识别结果:\n" + json.dumps(turn.skill_rec, ensure_ascii=False,
                                                   indent=4) + "\n工具调用结果:\n" + "\n".join(
        [tool_res.tool_call.__str__() for tool_res in turn.tool_res]) + "\n" + "faqs:\n" + json.dumps(turn.faq_res,
                                                                                                     ensure_ascii=False,
                                                                                                     indent=4)
    cur_user_input = messages[-1][CONTENT][0].text + "\n"
    return Message(USER, history_str + "\n" + "\n## Input:\nuser:" + cur_user_input + cur_turn_info + "\n## Reply:\n")


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
        final_messages = []

        messages = copy.deepcopy(messages)

        final_messages.insert(0, Message(SYSTEM, PROMPT_TEMPLATE[lang]))

        # 构建对话历史字符串，包括最近最近三次用户和assistant的对话记录
        # history_str = ""
        # for turn in kwargs["sessions"].turns[-3:]:
        #     history_str += "user:" + turn.user_input + "\n" + "assistant:" + turn.assistant_output + "\n"

        # 如果最后一条消息是用户消息，则在消息前添加对话历史和解析结果提示
        final_messages.append(
            _build_prompt(messages=messages, sessions=kwargs.get("sessions"), turn=kwargs.get("turn")))
        logger.info("-" * 20)
        logger.info(f"final_messages: {final_messages}")
        # 调用大语言模型处理消息列表，并返回处理结果
        output_stream = self._call_llm(messages=final_messages)
        return output_stream
