import copy
import json
import re
import time
import traceback
from typing import Dict, Iterator, List, Optional, Union, Tuple
import os
import json5

from qwen_agent import Agent
from qwen_agent.llm.base import BaseChatModel
from qwen_agent.llm.schema import CONTENT, DEFAULT_SYSTEM_MESSAGE, Message, FUNCTION, Session, Turn, SKILL_REC, \
    FunctionCall, ToolResponse, ToolCall, ASSISTANT
from qwen_agent.agents import SkillRecognizer
from .faq_agent import FAQAgent
from .summarize import Summarizer

from qwen_agent.tools import BaseTool
from ...utils.str_processing import rm_json_md, stream_string_by_chunk

DEFAULT_NAME = '财富顾问'
DEFAULT_DESC = '我是一个财富顾问，我能了解你的投资情况，帮你分析当下市场状况、产品信息。'
RESOURCE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resource")


class WealthyConsultant(Agent):
    """This is an agent for doc QA."""

    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
                 name: Optional[str] = DEFAULT_NAME,
                 description: Optional[str] = DEFAULT_DESC):
        super().__init__(function_list=function_list,
                         llm=llm,
                         system_message=system_message,
                         name=name,
                         description=description)
        with open(os.path.join(RESOURCE_PATH, "model_config.json"), "r", encoding="utf-8") as f:
            self.model_cfg = json.load(f)
        self.skill_rec = SkillRecognizer(llm=self.model_cfg["tool_llm_cfg"])
        self.session = Session(turns=[])

        self.summarizer = Summarizer(llm=self.model_cfg["summarize_llm_cfg"])
        self.faq_searcher = FAQAgent(function_list=["faq_embedding"], llm=self.model_cfg["tool_llm_cfg"])

    def _run(self, messages: List[Message], lang: str = 'zh', **kwargs) -> Iterator[List[Message]]:
        if messages[-1].content[0].text.strip() == '':
            raise ValueError("请输入有关信息")
        # New Turn, add user message to session
        turn = Turn(user_input=messages[-1].content[0].text)
        yield [Message(role=ASSISTANT, content="[DEBUG]正在识别已有 FAQ ...")]

        faq_start = time.time()
        faq_res = list(self.faq_searcher.run(messages=messages, sessions=self.session, lang=lang))[-1][0].content
        turn.faq_res = faq_res
        if faq_res != "[]":
            yield [Message(role=ASSISTANT, content="```json\n" + faq_res[1:-1] + "\n```", name="FAQ")]

        print(f"faq cost :{time.time() - faq_start}")

        # call skill recognizer
        yield [Message(role=ASSISTANT, content="[DEBUG]正在识别工具...")]
        skill_start = time.time()
        skill_res = list(self.skill_rec.run(messages=messages, sessions=self.session, function_map=self.function_map,
                                            lang=lang))[-1][0]
        skill_res_text = rm_json_md(skill_res.content)
        if "```" not in skill_res_text: skill_res_text = "```json\n" + skill_res_text + "\n```"
        for rsp in stream_string_by_chunk(skill_res_text):
            yield [Message(role=ASSISTANT, content=rsp, name="Skill Recognize")]
        turn.skill_rec = skill_res_text
        print(f"skill cost :{time.time() - skill_start}")

        # detect and call tools
        use_tool, tool_name, tool_args, _ = self._detect_tool(skill_res)
        tool_start = time.time()
        tool_response = ToolResponse(reply="", tool_call=None)

        if use_tool and tool_name in self.function_map:
            yield [Message(role=ASSISTANT, content=f"[DEBUG]正在调用 {tool_name} 工具...")]
            tool_response = self._call_tool(tool_name, tool_args, messages=messages, llm=self.llm,
                                            session=self.session,
                                            **kwargs)

        # add tools result to session
        turn.tool_res = tool_response
        print(f"tool cost :{time.time() - tool_start}")
        if tool_response.tool_call is not None:
            for rsp in stream_string_by_chunk("```json\n" + tool_response.tool_call.__str__() + "\n```"):
                yield [
                    Message(role=ASSISTANT, content=rsp, name="ToolCall")]
        summarize_start = time.time()
        # if tool_response.tool_call is None:
        #     turn.assistant_output = tool_response.reply
        #     for t in stream_string_by_chunk(tool_response.reply):
        #         yield [Message(role="assistant", content=t)]
        # else:
        response = []
        for rsp in self.summarizer.run(messages=messages, sessions=self.session, turn=turn):
            response += rsp
            yield rsp
        turn.assistant_output = response[-1].content
        print(f"summarize cost :{time.time() - summarize_start}")
        self.session.add_turn(turn)
        print(self.session.__repr__())

    def _detect_tool(self, message: Message) -> Tuple[bool, str, dict, str]:
        """A built-in tool call detection for func_call format message.

        Args:
            message: one message generated by LLM.

        Returns:
            Need to call tool or not, tool name, tool args, text replies.
        """
        # if isinstance(message, dict):
        #     if "function_call" in message:
        #         # todo: process multi function calls
        #         func = message["function_call"][0]
        #         message = Message(
        #             function_call=FunctionCall(name=func["name"], arguments=json.dumps(func["parameters"])),
        #             role="assistant", content="")
        content = message.content
        if "```" in content:
            content = rm_json_md(content)
        func = {"function_call": [{"name": None, "parameters": None}]}
        try:
            func = json5.loads(content)
        except Exception as e:
            print(f"解析json失败: {e} {content}")
        if len(func["function_call"]) != 0:
            func_name, func_args = func["function_call"][0]["name"], func["function_call"][0]["parameters"]
        else:
            func_name, func_args = None, {}
        text = message.content
        if not text:
            text = ''

        return (func_name is not None), func_name, func_args, text
