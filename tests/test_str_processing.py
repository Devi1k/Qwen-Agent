from qwen_agent.utils.str_processing import rm_json_md

test_query = """
```json
{"test":"content"}
```
"""


def test_rm_json_md():
    print(rm_json_md(test_query))
