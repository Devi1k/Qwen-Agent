from qwen_agent.agents import SkillRecognizer
from qwen_agent.llm.schema import Session


def test_faq_agent():
    llm_cfg = {'model': 'qwen-max',
               'api_key': 'sk-22a3f18de8c840d79d3d16f821c9a160',
               'model_server': 'dashscope',
               'generate_cfg':
                   {'temperature': 0.1, 'top_p': 0.7, 'max_tokens': 1024}
               }
    agent = SkillRecognizer(llm=llm_cfg, function_list=["产品查询", "产品推荐", "持仓查询", "提交订单"])
    messages = [{
        'role': 'user',
        'content': [{
            'text': '请帮我下单一个基金，基金名称是中证指数基金，基金份额是10000'
        }]
    }]
    for rsp in agent.run(messages, function_map=agent.function_map, sessions=Session(turns=[])):
        print(rsp)
