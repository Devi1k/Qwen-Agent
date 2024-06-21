import os
from pprint import pprint

from qwen_agent.agents import WealthyConsultant
from qwen_agent.gui import WebUI

tool_llm_cfg = {
    'model': 'qwen1.5-14b-chat',
    'model_server': 'dashscope',
    "api_key": "sk-22a3f18de8c840d79d3d16f821c9a160",
    'generate_cfg': {
        'temperature': 0.01
    }
}

tool_list = ["产品查询", "产品推荐"]


def wealthy_consultant():
    bot = WealthyConsultant(llm=tool_llm_cfg,
                            name="财顾小信",
                            description="我是一个财富顾问，我能了解你的投资情况，帮你分析当下市场状况、产品信息。",
                            function_list=tool_list)

    while True:
        query = input("请输入查询内容：")
        if query == "exit":
            break
        messages = [{'role': 'user', 'content': [{'text': query}]}]
        for rsp in bot.run(messages):
            pprint(rsp)


def app_gui():
    bot = WealthyConsultant(llm=tool_llm_cfg,
                            name="财顾小信",
                            description="我是一个财富顾问，我能了解你的投资情况，帮你分析当下市场状况、产品信息。",
                            function_list=tool_list)
    chatbot_config = {
        'verbose': True,
        'prompt.suggestions': [
            '华宝新兴最近能加仓吗',
            '安鑫稳评测一下这个产品',
            '安享惠金的投资风格是什么',
            '推荐几个低风险产品'
        ]
    }
    WebUI(
        bot,
        chatbot_config=chatbot_config,
    ).run(share=True)


if __name__ == '__main__':
    # wealthy_consultant()
    os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
    app_gui()
