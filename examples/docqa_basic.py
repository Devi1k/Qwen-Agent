import os

from qwen_agent.agents.doc_qa import BasicDocQA
from qwen_agent.gui import WebUI


def doc_qa_test():
    bot = BasicDocQA(llm={'model': 'qwen-plus'})
    messages = [{'role': 'user', 'content': [{'text': '介绍图一'}, {'file': 'https://arxiv.org/pdf/1706.03762.pdf'}]}]
    for rsp in bot.run(messages):
        print(rsp)


def app_gui():
    # Define the agent
    bot = BasicDocQA(llm={'model': 'qwen-plus', 'model_server': 'dashscope', "api_key": "sk-22a3f18de8c840d79d3d16f821c9a160"})
    chatbot_config = {
        'prompt.suggestions': [
            {
                'text': '介绍图一'
            },
            {
                'text': '第二章第一句话是什么？'
            },
        ],
    }
    WebUI(bot, chatbot_config=chatbot_config).run()


if __name__ == '__main__':
    os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

    # test()
    app_gui()
