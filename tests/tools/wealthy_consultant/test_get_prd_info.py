import pytest

from qwen_agent.tools import GetProductInfo

params = {
    "case1": {
        'product_name': "华安",
        'field_type': "产品评测"
    },
    "case2": {
        'field_type': "加减仓市场分析",
        'product_id': "011696"
    }
}


@pytest.mark.parametrize(
    'case', [v for k, v in params.items()]
)
def test_get_account_info(case):
    tool = GetProductInfo()
    print(tool.function)
    print(tool.call(case))
