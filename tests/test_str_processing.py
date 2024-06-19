from pprint import pprint

import json5
import pytest

from qwen_agent.utils.str_processing import rm_json_md


@pytest.mark.parametrize('test_query',
                         ["\n```json\n{\n    \"faqs\": []\n}\n```\n", "{\n    \"faqs\": []\n}我从中抽取了上文内容",
                          "```\n{\n    \"faqs\": []\n}\n```\n",
                          "```json\n{\n\"是否需要澄清\": \"是\",\"链接结果\": [{\"type\": \"基金\",\"name\": \"华宝新兴产业混合\"}]}\n```"])
def test_rm_json_md(test_query):
    # pprint(test_query)
    print(json5.loads(rm_json_md(test_query)))
