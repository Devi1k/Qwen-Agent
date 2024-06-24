import pytest

from qwen_agent.tools import Recommend

params = {
    "case1": {'product_style': "", 'risk_level': "低风险", 'investment_sector': ""},
    "case2": {'product_style': "科技主颗，偏小盘", 'risk_level': "", 'investment_sector': ""},
    "case3": {'product_style': "", 'risk_level': "", 'investment_sector': "科技"},
}


@pytest.mark.parametrize(
    'params', [v for v in params.values()]
)
def test_get_recommend(params):
    tool = Recommend()
    print(tool.call(params))
