from qwen_agent.agents import WealthyConsultant


def wealthy_consultant():
    llm_cfg = {'model': 'qwen2-72b-instruct', 'model_server': 'dashscope', "api_key": "sk-22a3f18de8c840d79d3d16f821c9a160"}

    bot = WealthyConsultant(llm=llm_cfg, function_list=["get_account_info"])

    while True:
        query = input("请输入查询内容：")
        if query == "exit":
            break
        messages = [{'role': 'user', 'content': [{'text': query}]}]
        print(list(bot.run(messages)))


if __name__ == '__main__':
    wealthy_consultant()
