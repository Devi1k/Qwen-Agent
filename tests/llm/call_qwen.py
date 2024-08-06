from http import HTTPStatus

import dashscope
import qianfan


# dashscope.api_key = "sk-22a3f18de8c840d79d3d16f821c9a160"
def call_with_messages():
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': '请介绍一下通义千问'}]

    response = dashscope.Generation.call(
        dashscope.Generation.Models.qwen_turbo,
        messages=messages,
        result_format='message',  # 将返回结果格式设置为 message
    )
    if response.status_code == HTTPStatus.OK:
        print(response)
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))


def call_qianfan():
    import os
    os.environ["QIANFAN_ACCESS_KEY"] = "ALTAKt7kVm6qg5eZQaOlVUR3l0"
    os.environ["QIANFAN_SECRET_KEY"] = "5a2e43049458416a83e51497d1ebdaef"

    # 指定特定模型
    resp = qianfan.ChatCompletion().do(model="ERNIE-3.5-8K", messages=[{
        "role": "user",
        "content": "你是财富顾问小信。你好，你是谁"
    },
        {
            "role": "assistant",
            "content": "你好！我是财富顾问小信，专注于为客户提供专业的财富管理建议和解决方案。请问你是对理财、投资、保险还是其他财富管理方面有所需求呢？我很乐意为你提供专业的帮助和建议。"
        },
        {
            "role": "user", "content": "大盘 K 线怎么看"
        }], stream=True)

    for rsp in resp:
        print(rsp.body.get("result"))
    print(rsp)


if __name__ == '__main__':
    call_qianfan()
