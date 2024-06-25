import pytest

from qwen_agent.tools import GetProductInfo

params = {
    # "case1": {
    #     'product_name': "南方",
    #     'field_type': "产品评测",
    #     "product_type": "基金"
    # },
    # "case2": {
    #     'field_type': "加减仓市场分析",
    #     'product_id': "011696"
    # },
    # "case3": {
    #     "investment_sector": "科技",
    #     "product_type": "基金"
    # },
    "case4": {
        "product_name": "中银收益混合A,汇添富上证综合指数A"
    }
}


@pytest.mark.parametrize(
    'case', [v for k, v in params.items()]
)
def test_get_account_info(case):
    tool = GetProductInfo()
    print(tool.call(case))
