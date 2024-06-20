from typing import List, Literal, Optional, Tuple, Union, Dict

from pydantic import BaseModel, field_validator, model_validator

DEFAULT_SYSTEM_MESSAGE = 'You are a helpful assistant.'

ROLE = 'role'
CONTENT = 'content'
NAME = 'name'

SYSTEM = 'system'
USER = 'user'
ASSISTANT = 'assistant'
FUNCTION = 'function'
SKILL_REC = 'skill_rec'

FILE = 'file'
IMAGE = 'image'


class BaseModelCompatibleDict(BaseModel):

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def model_dump(self, **kwargs):
        return super().model_dump(exclude_none=True, **kwargs)

    def model_dump_json(self, **kwargs):
        return super().model_dump_json(exclude_none=True, **kwargs)

    def get(self, key, default=None):
        try:
            value = getattr(self, key)
            if value:
                return value
            else:
                return default
        except AttributeError:
            return default

    def __str__(self):
        return f'{self.model_dump()}'


class FunctionCall(BaseModelCompatibleDict):
    name: str
    arguments: str

    def __init__(self, name: str, arguments: str):
        super().__init__(name=name, arguments=arguments)

    def __repr__(self):
        return f'FunctionCall({self.model_dump_json()})'


class ContentItem(BaseModelCompatibleDict):
    text: Optional[str] = None
    image: Optional[str] = None
    file: Optional[str] = None

    def __init__(self, text: Optional[str] = None, image: Optional[str] = None, file: Optional[str] = None):
        super().__init__(text=text, image=image, file=file)

    @model_validator(mode='after')
    def check_exclusivity(self):
        provided_fields = 0
        if self.text is not None:
            provided_fields += 1
        if self.image:
            provided_fields += 1
        if self.file:
            provided_fields += 1

        if provided_fields != 1:
            raise ValueError("Exactly one of 'text', 'image', or 'file' must be provided.")
        return self

    def __repr__(self):
        return f'ContentItem({self.model_dump()})'

    def get_type_and_value(self) -> Tuple[Literal['text', 'image', 'file'], str]:
        (t, v), = self.model_dump().items()
        assert t in ('text', 'image', 'file')
        return t, v

    @property
    def type(self) -> Literal['text', 'image', 'file']:
        t, v = self.get_type_and_value()
        return t

    @property
    def value(self) -> str:
        t, v = self.get_type_and_value()
        return v


class Message(BaseModelCompatibleDict):
    role: str
    content: Union[str, List[ContentItem]]
    name: Optional[str] = None
    function_call: Optional[FunctionCall] = None

    def __init__(self,
                 role: str,
                 content: Optional[Union[str, List[ContentItem]]],
                 name: Optional[str] = None,
                 function_call: Optional[FunctionCall] = None,
                 **kwargs):
        if content is None:
            content = ''
        super().__init__(role=role, content=content, name=name, function_call=function_call)

    def __repr__(self):
        return f'Message({self.model_dump()})'

    @field_validator('role')
    def role_checker(cls, value: str) -> str:
        if value not in [USER, ASSISTANT, SYSTEM, FUNCTION]:
            raise ValueError(f'{value} must be one of {",".join([USER, ASSISTANT, SYSTEM, FUNCTION])}')
        return value


class ToolCall(BaseModelCompatibleDict):
    """
        action: 工具名称
        description: 工具描述
        action_input: 工具输入
        observation: 工具输出
    """
    action: str
    description: str
    action_input: Dict
    observation: List[Dict]

    def __init__(self, action: str, description: str, action_input: Dict, observation: List[Dict]):
        super().__init__(action=action, description=description, action_input=action_input, observation=observation)

    def __repr__(self):
        return f'{self.model_dump()}'

    def __str__(self):
        return f'{self.model_dump_json(indent=4)}'


class ToolResponse(BaseModelCompatibleDict):
    reply: str
    tool_call: Optional[ToolCall]
    card: Optional[Dict]

    def __init__(self, reply: str, tool_call: Optional[ToolCall] = None, card: Optional[Dict] = None):
        super().__init__(reply=reply, tool_call=tool_call, card=card)


class Turn(BaseModelCompatibleDict):
    user_input: str
    assistant_output: Optional[str]
    skill_rec: Optional[str] = None
    tool_res: Optional[List[ToolResponse]] = None
    faq_res: Optional[str] = None

    def __init__(self,
                 user_input: Optional[str] = None,
                 assistant_output: Optional[str] = None,
                 skill_rec: Optional[str] = None,
                 tool_res: Optional[List[ToolResponse]] = None,
                 faq_res: Optional[str] = None,
                 **kwargs):
        super().__init__(user_input=user_input, assistant_output=assistant_output, skill_rec=skill_rec,
                         tool_res=tool_res, faq_res=faq_res)

    def __repr__(self):
        return f'{self.model_dump()}'

    def __str__(self):
        return f'{self.model_dump_json(indent=4)}'


class Session:
    turns: List[Turn]

    def __init__(self, turns: List[Turn]):
        self.turns = turns

    def add_turn(self, turn: Turn):
        self.turns.append(turn)

    def get_whole_history(self):
        history_str = "## History: \n"
        for turn in self.turns[-3:]:
            user = "user:" + turn.user_input + "\n"
            faq_res = "faq result:\n" + turn.faq_res + "\n"
            tool_res = "tool result:" + "\n"
            for t_r in turn.tool_res:
                tool_res += t_r.tool_call.__str__() if t_r.tool_call is not None else t_r.reply + "\n"
            assistant_output = "assistant:" + turn.assistant_output + "\n"
            history_str += user + faq_res + tool_res + assistant_output
        return history_str

    def __repr__(self):
        return [f"turn {index + 1}:" + turn.__str__() + "\n" for index, turn in enumerate(self.turns)]
