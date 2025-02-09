"""
Jinja2 template for Deepseek chat generation.
"""

import textwrap


# Largely taken from https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/tokenizer_config.json
# However, the built-in chat template splits the content at "</think>", which makes iterative generation very
# difficult. Not sure why that was included, but it works without it.
template = (
    textwrap.dedent(
        """
    {% if not add_generation_prompt is defined %}
    {% set add_generation_prompt = false %}
    {% endif %}
    {% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='', is_first_sp=true) %}
    {%- for message in messages %}
    {%- if message['role'] == 'system' %}
    {%- if ns.is_first_sp %}
    {% set ns.system_prompt = ns.system_prompt + message['content'] %}
    {% set ns.is_first_sp = false %}
    {%- else %}
    {% set ns.system_prompt = ns.system_prompt + '\\n\\n' + message['content'] %}
    {%- endif %}
    {%- endif %}
    {%- endfor %}
    {{ bos_token }}
    {{ ns.system_prompt }}
    {%- for message in messages %}
    {%- if message['role'] == 'user' %}
    {%- set ns.is_tool = false -%}
    {{'<｜User｜>' + message['content']}}
    {%- endif %}
    {%- if message['role'] == 'assistant' and 'tool_calls' in message %}
    {%- set ns.is_tool = false -%}
    {%- for tool in message['tool_calls'] %}
    {%- if not ns.is_first %}
    {%- if message['content'] is none %}
    {{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}
    {%- else %}{{'<｜Assistant｜>' + message['content'] + '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}
    {%- endif %}
    {%- set ns.is_first = true -%}
    {%- else %}
    {{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}
    {%- endif %}
    {%- endfor %}
    {{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}
    {%- endif %}{%- if message['role'] == 'assistant' and 'tool_calls' not in message %}
    {%- if ns.is_tool %}
    {{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}
    {%- set ns.is_tool = false -%}
    {%- else %}
    {% set content = message['content'] %}
    {{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}
    {%- endif %}
    {%- endif %}
    {%- if message['role'] == 'tool' %}
    {%- set ns.is_tool = true -%}
    {%- if ns.is_output_first %}
    {{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}
    {%- set ns.is_output_first = false %}
    {%- else %}
    {{'<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}
    {%- endif %}
    {%- endif %}
    {%- endfor -%}
    {% if ns.is_tool %}
    {{'<｜tool▁outputs▁end｜>'}}
    {% endif %}
    {% if add_generation_prompt and not ns.is_tool %}
    {{'<｜Assistant｜>'}}
    {% endif %}
"""
    )
    .replace("\n", "")
    .strip()
)
