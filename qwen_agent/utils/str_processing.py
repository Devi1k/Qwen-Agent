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
    """
    去除文本中的json格式
    """
    json_patterns = [
        r'```(?:JSON|json|Json)\s*(.*?)```',
        r'```\s*(.*?)```',
        r'\{.*?\}'
    ]
    matches_findall = [match for pattern in json_patterns for match in re.findall(pattern, text)]
    if matches_findall:
        return matches_findall[0]
    return text


def stream_string_by_chunk(s, chunk_size=6):
    res = []
    tmp = ""
    for i in range(0, len(s), chunk_size):
        tmp += s[i:i + chunk_size]
        res.append(tmp)
    return res
