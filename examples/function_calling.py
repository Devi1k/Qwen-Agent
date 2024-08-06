# Reference: https://platform.openai.com/docs/guides/function-calling
import json
import os

from qwen_agent.llm import get_chat_model


# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit='fahrenheit'):
    """Get the current weather in a given location"""
    if 'tokyo' in location.lower():
        return json.dumps({'location': 'Tokyo', 'temperature': '10', 'unit': 'celsius'})
    elif 'san francisco' in location.lower():
        return json.dumps({'location': 'San Francisco', 'temperature': '72', 'unit': 'fahrenheit'})
    elif 'paris' in location.lower():
        return json.dumps({'location': 'Paris', 'temperature': '22', 'unit': 'celsius'})
    else:
        return json.dumps({'location': location, 'temperature': 'unknown'})

def get_product_info(product_type, field_type, product_name, product_id, product_manager):
    """Get the product information"""
    return json.dumps({'product_type': product_type, 'field_type': field_type, 'product_name': product_name, 'product_id': product_id, 'product_manager': product_manager})

def recommend(product_style, risk_level, investment_sector):
    """Recommend relevant products based on specified criteria"""
    return json.dumps({'recommendation': 'Fund A, Fund B, Fund C'})
def test():
    llm = get_chat_model({
        # Use the model service provided by DashScope:
        'model': 'qwen-max',
        'model_server': 'dashscope',
        'api_key': os.getenv('DASHSCOPE_API_KEY'),

        # Use the model service provided by Together.AI:
        # 'model': 'Qwen/Qwen1.5-14B-Chat',
        # 'model_server': 'https://api.together.xyz',  # api_base
        # 'api_key': os.getenv('TOGETHER_API_KEY'),

        # Use your own model service compatible with OpenAI API:
        # 'model': 'Qwen/Qwen1.5-72B-Chat',
        # 'model_server': 'http://localhost:8000/v1',  # api_base
        # 'api_key': 'EMPTY',
    })

    # Step 1: send the conversation and available functions to the model
    messages = [{'role': 'user', 'content': "查询安盈象这个产品，并推荐同风格基金"}]
    functions = [{
        'name': 'get_current_weather',
        'description': 'Get the current weather in a given location',
        'parameters': {
            'type': 'object',
            'properties': {
                'location': {
                    'type': 'string',
                    'description': 'The city and state, e.g. San Francisco, CA',
                },
                'unit': {
                    'type': 'string',
                    'enum': ['celsius', 'fahrenheit']
                },
            },
            'required': ['location'],
        },
    },
        {
            'name': 'get_product_info',
            'description': 'Get the product information',
            'parameters': {
                'type': 'object',
                'properties': {
                    'product_type': {
                        'type': 'string',
                        'enum': ['基金', '理财'],
                        'description': '基金/理财'
                    },
                    'field_type': {
                        'type': 'string',
                        'enum': ['产品评测', '加减仓市场分析'],
                        'description': '产品评测/加减仓市场分析'
                    },
                    'product_name': {
                        'type': 'string',
                        'description': '基金/理财产品的名称'
                    },
                    'product_id': {
                        'type': 'string',
                        'description': '基金/理财产品的产品代码'
                    },
                    'product_manager': {
                        'type': 'string',
                    }
                }
            }
        },{
            'name': 'recommend',
            'description': 'This tool is designed to recommend relevant products based on specified criteria.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'product_style': {
                        'type': 'string',
                        'enum': ['均衡', '偏大盘', '中波动固收+', '偏中短债', '偏成长', '大小盘平衡', '偏中短债', '大小盘平衡 ',
                                 '科技主题', '偏小盘', '医药主题', '大小盘平衡'],
                        'description': '产品风格'
                    },
                    'risk_level': {
                        'type': 'string',
                        'enum': ['低风险', '中低风险', '中风险', '中高风险', '高风险'],
                        'description': '风险等级'
                    },
                    'investment_sector': {
                        'type': 'string',
                        'enum': []
                    }
                }
            }
        }]

    print('# Assistant Response 1:')
    responses = []
    for responses in llm.chat(messages=messages, functions=functions, stream=True):
        print(responses)

    messages.extend(responses)  # extend conversation with assistant's reply

    # Step 2: check if the model wanted to call a function
    last_response = messages[-1]
    if last_response.get('function_call', None):

        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            'get_current_weather': get_current_weather,
            'get_product_info': get_product_info,
            'recommend': recommend,
        }
        # only one function in this example, but you can have multiple
        function_name = last_response['function_call']['name']
        function_to_call = available_functions[function_name]
        function_args = json.loads(last_response['function_call']['arguments'])
        function_response = function_to_call(
            location=function_args.get('location'),
            unit=function_args.get('unit'),
        )
        print('# Function Response:')
        print(function_response)

        # Step 4: send the info for each function call and function response to the model
        messages.append({
            'role': 'function',
            'name': function_name,
            'content': function_response,
        })  # extend conversation with function response

        print('# Assistant Response 2:')
        for responses in llm.chat(
                messages=messages,
                functions=functions,
                stream=True,
        ):  # get a new response from the model where it can see the function response
            print(responses)


if __name__ == '__main__':
    test()
