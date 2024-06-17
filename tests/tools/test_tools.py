import json

import pytest

from qwen_agent.tools import AmapWeather, CodeInterpreter, ImageGen, Retrieval, Storage, GetAccountInfo, Recommend


# [NOTE] 不带“市”会出错
@pytest.mark.parametrize('params', [json.dumps({'location': '北京市'}), {'location': '杭州市'}])
def test_amap_weather(params):
    tool = AmapWeather()
    tool.call(params)


params = {
    "case1": {'cstno': 62121206326456862, 'ctfno': 11010120912085089124, 'cstname': "张三"}
}


@pytest.mark.parametrize(
    'params', params.values()
)
def test_get_account_info(params):
    tool = GetAccountInfo()
    print(tool.function)
    tool.call(params)


params = {
    "case1": {'product_style': "", 'risk_level': "", 'investment_sector': "科技"}
}


@pytest.mark.parametrize(
    'params', params.values()
)
def test_get_recommend(params):
    tool = Recommend()
    print(tool.function)
    tool.call(params)


def test_code_interpreter():
    tool = CodeInterpreter()
    tool.call("print('hello qwen')")


def test_image_gen():
    tool = ImageGen()
    tool.call({'prompt': 'a dog'})


def test_retrieval():
    tool = Retrieval({"rag_searchers": ["vector_search"]})
    tool.call({
        'query': 'Who are the authors of this paper?',
        'files': ['https://qianwen-res.oss-cn-beijing.aliyuncs.com/QWEN_TECHNICAL_REPORT.pdf']
    })


@pytest.mark.parametrize('operate', ['put'])
def test_storage_put(operate):
    tool = Storage()
    tool.call({'operate': operate, 'key': '345/456/11', 'value': 'hello'})

    tool.call({'operate': operate, 'key': '/345/456/12', 'value': 'hello'})


@pytest.mark.parametrize('operate', ['scan'])
def test_storage_scan(operate):
    tool = Storage()
    tool.call({'operate': operate, 'key': '345/456/'})

    tool.call({'operate': operate, 'key': '/345/456'})


@pytest.mark.parametrize('operate', ['get', 'delete'])
def test_storage_get_delete(operate):
    tool = Storage()
    tool.call({'operate': operate, 'key': '345/456/11'})

    tool.call({'operate': operate, 'key': '/345/456/12'})
