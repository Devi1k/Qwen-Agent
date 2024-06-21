from qwen_agent.agents import FAQAgent


def test_faq_agent():
    llm_cfg = {'model': 'qwen-max',
               'api_key': 'sk-22a3f18de8c840d79d3d16f821c9a160',
               'model_server': 'dashscope',
               'generate_cfg':
                   {'temperature': 0.1, 'top_p': 0.7, 'max_tokens': 1024}
               }
    agent = FAQAgent(llm=llm_cfg, function_list=["faq_embedding"])
    messages = [{
        'role': 'user',
        'content': [{
            'text': '什么是 CPI'
        }]
    }]
    for rsp in agent.run(messages, full_article=True):
        print(rsp)
