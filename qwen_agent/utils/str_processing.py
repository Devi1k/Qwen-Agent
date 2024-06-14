import re

from qwen_agent.utils.utils import has_chinese_chars


def rm_newlines(text):
    if text.endswith('-\n'):
        text = text[:-2]
        return text.strip()
    rep_c = ' '
    if has_chinese_chars(text):
        rep_c = ''
    text = re.sub(r'(?<=[^\.。:：\d])\n', rep_c, text)
    return text.strip()


def rm_cid(text):
    text = re.sub(r'\(cid:\d+\)', '', text)
    return text


def rm_hexadecimal(text):
    text = re.sub(r'[0-9A-Fa-f]{21,}', '', text)
    return text


def rm_continuous_placeholders(text):
    text = re.sub(r'[.\- —。_*]{7,}', '\t', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def rm_json_md(text):
    pattern = r'```(?:JSON|json|Json)\s*(.*?)```'
    try:
        clarify_llm_res = re.findall(pattern, text, re.DOTALL)[0]
    except IndexError as e:
        pattern_b = r'```\s*(.*?)```'
        clarify_llm_res = re.findall(pattern_b, text, re.DOTALL)
        if not clarify_llm_res:
            print(f"文本：{text} 格式错误 {e} 没有匹配 json 内容")
            return text
        else:
            return clarify_llm_res[0]
    return clarify_llm_res
