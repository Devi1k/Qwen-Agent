import pytest

from qwen_agent.tools import SubmitOrder

params = {
    "case1": {'product_name': "国联安鑫稳3个月持有混合C", "purchase_share": "10000"},
    "case2": {'product_name': "南方大数据100指数A", "purchase_share": "300000"},
}


@pytest.mark.parametrize(
    'params', [v for v in params.values()]
)
def test_get_recommend(params):
    tool = SubmitOrder()
    print(tool.call(params))
