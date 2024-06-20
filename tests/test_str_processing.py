import json5
import pytest

from qwen_agent.utils.str_processing import rm_json_md


@pytest.mark.parametrize('test_query',
                         ["\n```json\n{\n    \"faqs\": []\n}\n```\n", "{\n    \"faqs\": []\n}我从中抽取了上文内容",
                          "```\n{\n    \"faqs\": []\n}\n```\n",
                          "```json\n{\n\"是否需要澄清\": \"是\",\"链接结果\": [{\"type\": \"基金\",\"name\": \"华宝新兴产业混合\"}]}\n```",
                          "{\"thought\": \"用户要求对比中银和汇添富的产品，但之前推荐的汇添富是指数基金，而中银收益混合A是主动管理的混合型基金。为了提供准确的对比，我需要查询汇添富的具体产品信息，可能是同类的混合型基金。\", \"function_call\": [{\"name\": \"产品查询\", \"parameters\": {\"product_name\": \"汇添富\", \"product_type\": \"基金\", \"field_type\": \"产品评测,加减仓市场分析\"}}]}\""])
def test_rm_json_md(test_query):
    # pprint(test_query)
    print(json5.loads(rm_json_md(test_query)))
