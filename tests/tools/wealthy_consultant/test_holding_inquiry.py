from qwen_agent.tools import HoldingsInquiry


def test_holding_inquiry():
    params = {"cstno": 120106199904210011}
    tool = HoldingsInquiry()
    print(tool.call(params))
