### 背景：
我们已经设计了一版比较好的 prompt，还是会遇到不少不够遵循指令的情况，比如：
- 1）不合法的 json，如丢失的逗号、不匹配的引号；
- 2）错误的字段，如“情绪”变成了“情感”；
- 3）冗余的回复，如以“以下是情绪识别结果：” 开头、以 “整体上，该内容表达了......” 结尾。对于异常格式，一般是根据实际的 case 设计一些后处理方法进一步规范回答的格式，但这也非常耗费精力。

### 1. JSON Schema 约束模型输出结构
核心原理： guided decoding（引导解码）。在模型生成每个 token 时，根据 JSON Schema 实时限制下一个 token 的合法取值空间，从而强制输出结构合法的 JSON。

JSON Schema 提供结构约束
```python
在 extra_body 中可填并且比较常用的字段有：
guided_choice 输出将会是其中一个选项。
guided_regex 输出将遵循正则表达式模式。
guided_json 输出将遵循 JSON 架构(pydantic的json格式描述)。
guided_grammar 输出将遵循上下文无关文法。
```
模型生成过程中实时限制 token 选择，在解码（decoding）阶段，每个 token 的生成不是自由的，而是根据 JSON Schema 实时过滤允许的 token。
例如：
当模型刚生成 { 后，schema 要求下一个 key 必须是 "问题"，所以只允许生成 "问题" 这个 token。
当生成完 "问题": " 后，schema 要求接下来是字符串值，所以只允许生成合法字符串 token，直到遇到 "。

| 后端框架                         | 实现方式                                                                                                    |
| ---------------------------- | ------------------------------------------------------------------------------------------------------- |
| **vLLM**                     | 使用 [`outlines`](https://github.com/outlines-dev/outlines) 或 `lm-format-enforcer`，在 logits 阶段屏蔽非法 token。 |
| **HuggingFace Transformers** | 使用 `transformers` 的 `LogitsProcessor`，如 `JSONLogitsProcessor`。                                          |
| **OpenAI API（私有）**           | 内部实现不公开，但原理类似，通过 schema 限制 token 采样空间。                                                                  |


```python
from pydantic import BaseModel
class Topic(BaseModel):
    问题: str
    答案: str

completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    extra_body={"guided_json": Topic.model_json_schema()},
)
print(completion.choices[0].message.content)

Topic.model_json_schema()
会生成如下 JSON Schema：
{
  "type": "object",
  "properties": {
    "问题": {"type": "string"},
    "答案": {"type": "string"}
  },
  "required": ["问题", "答案"]
}

```


### 2. 使用json-repair包
原理：这个包能智能地处理各种常见错误
链接：https://github.com/mangiucugna/json_repair/
`json_repair` 的 `repair_json` 函数不仅能处理 dict 格式，也能处理 array/list 格式。根据官方文档，它支持修复以下问题：
- **JSON 数组和对象** - 修复不完整或损坏的数组/对象，添加必要的元素（如逗号、括号）
- **单引号转双引号** - 自动将 Python 风格的单引号转换为标准 JSON 双引号
- **缺少引号的键名** - 自动添加引号
- **尾部多余逗号** - 智能移除
- **未闭合的括号** - 自动补全

缺点：返回结果里没有json格式的数据，就会失效
举例：
"好的，这是您要的JSON：\n{'user': 'Alex', 'id': 123}\n希望对您有帮助！"为例
这种一般是因为前后有多余的字，无法直接使用json.load进行转换，因为需要使用json-repair。


```python
from json_repair import repair_json
llm_output_string = "好的，这是您要的JSON：\n[{'user': 'Alex', 'id': 123},{'user': 'qyc', 'id': 234}]\n希望对您有帮助！"
repaired_string = repair_json(llm_output_string)
print(repaired_string, type(repaired_string))
import json
data = json.loads(repaired_string)
print(data,type(data))
# 输出： [{"user": "Alex", "id": 123}, {"user": "qyc", "id": 234}] <class 'str'>
# [{'user': 'Alex', 'id': 123}, {'user': 'qyc', 'id': 234}] <class 'list'>


from json_repair import repair_json, loads
# 示例1: 修复 list 格式（单引号、尾部逗号）
bad_list = "['apple', 'banana', 'cherry',]"
fixed = repair_json(bad_list)
print(fixed)  # 输出: ["apple", "banana", "cherry"]

# 示例2: 直接解析为 Python list
data = loads("['item1', 'item2']")
print(data)  # 输出: ['item1', 'item2'] (list类型)

# 示例3: 混合复杂结构（list包含dict）
complex_bad = "[{'name': 'Alice', 'age': 25}, {'name': 'Bob',}]"
result = loads(complex_bad)
print(result)  # 输出: [{'name': 'Alice', 'age': 25}, {'name': 'Bob'}]
```

### 3.ollama本身支持结果化输出
- 示例1
```python
from ollama import chat
from pydantic import BaseModel
# Ollama 近日就此问题推出重要更新（0.5.0+）
class Pet(BaseModel):
    name: str
    animal: str
    age: int
    color: str | None
    favorite_toy: str | None

class PetList(BaseModel):
    pets: list[Pet]

response = chat(
    messages=[
        {
            'role': 'user',
            'content': '''
            I have two pets.
            A cat named Luna who is 5 years old and loves playing with yarn. She is black and white.
            I also have a 2 year old black cat named Loki who loves tennis balls.
            '''
        },
    ],
    model="glm4:9b",
    format=PetList.model_json_schema(),
)
pets = PetList.model_validate_json(response.message.content)
print(pets)


```
- 示例2 使用 LangChain 的 ChatOllama
```python
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import json

# 初始化模型
llm = ChatOllama(
    model="gemma:7b",
    base_url="http://192.168.0.181:11434",
    format="json",
    temperature=0
)

# 构建消息
messages = [
    SystemMessage(content="请用中文回答我的提问。"),
    HumanMessage(
        content="请列出世界上 5 旅游城市，并且给出简短的描述? Respond using JSON"
    )
]

# 调用模型
chat_model_response = llm.invoke(messages)
print(json.loads(chat_model_response.content))

{
  "cities": [
    {
      "name": "伦敦",
      "description": "以其历史文化和金融中心而闻名，拥有丰富的博物馆和景点。"
    },
    {
      "name": "纽约",
      "description": "以其繁荣的都市生活和多元文化而闻名，拥有世界闻名的购物街和美食。"
    },
    {
      "name": "巴黎",
      "description": "以其浪漫的街道和历史文化而闻名，拥有著名的塔尔塔罗尼塔和埃菲尔塔。"
    },
    {
      "name": "罗马",
      "description": "以其古迹和历史文化而闻名，拥有著名的古罗马遗迹和维泰尔区。"
    },
    {
      "name": "迪拜",
      "description": "以其现代建筑和沙漠风情而闻名，拥有世界最大的购物中心和沙漠风情。"
    }
  ]
}
```
### 4.Instructor库
Instructor是一个基于Pydantic构建的Python库,旨在简化大语言模型(LLMs)结构化输出的处理过程。
- 响应模型: 使用Pydantic模型定义LLM输出的结构
- 重试管理: 轻松配置请求重试次数
- 验证: 使用Pydantic验证确保LLM响应符合预期
- 流式支持: 轻松处理列表和部分响应
- 灵活后端: 无缝集成OpenAI以外的多种LLM提供商

```bash
pip install -U instructor
```
- 示例1
```python
from openai import OpenAI
from pydantic import BaseModel
import instructor

# 定义数据结构
class CityDetail(BaseModel):
    name: str
    description: str

class Citylist(BaseModel):
    citys: list[CityDetail]

# 初始化客户端
client = instructor.from_openai(
    OpenAI(base_url="http://192.168.0.181:11434/v1", api_key="ollama"),
    mode=instructor.Mode.JSON
)

# 调用模型
resp = client.chat.completions.create(
    model="gemma:7b",
    response_model=Citylist,
    messages=[{
        "role": "user",
        "content": "请用中文回答我的提问。请列出世界上 5 旅游城市，并且给出简短的描述?"
    }],
)
print(resp)
citys=[
    CityDetail(name='纽约市', description='以高楼大厦和繁荣的商业活动而著称的金融中心'),
    CityDetail(name='伦敦', description='以历史建筑、文化和金融业而著称的欧洲都市'),
    CityDetail(name='巴黎', description='以其美丽的街区、美食和艺术而著称的法国首都'),
    CityDetail(name='罗馬', description='以其历史和文化而著称的意大利首都'),
    CityDetail(name='新加坡', description='以其现代建筑和繁荣的经济而著称的东南亚城市')
]
```
- 示例2
```python
from openai import OpenAI
from pydantic import BaseModel
import instructor

# 定义数据结构
class CityDetail(BaseModel):
    name: str
    country: str

# 初始化客户端
client = instructor.from_openai(
    OpenAI(base_url="http://192.168.0.181:11434/v1", api_key="ollama"),
    mode=instructor.Mode.JSON
)

# 调用模型
resp = client.chat.completions.create(
    model="gemma:7b",
    response_model=CityDetail,
    messages=[{"role": "user", "content": "中国的上海"}],
)

# 输出结果
print(type(resp), resp)        # <class '__main__.CityDetail'> name='上海' country='中国'
print(resp.country, resp.name) # 中国 上海
```
### 5. openai原生支持
- 示例1
```python
from openai import OpenAI
client = OpenAI(
    base_url="http://192.168.0.181:11434/v1",
    api_key="ollama",
)
resp = client.chat.completions.create(
    model="gemma:7b",
    messages=[
        {
            "role": "user",
            "content": "请列出世界上 5 旅游城市，并且给出简短的描述? Respond using JSON"
        }
    ],
    response_format={"type": "json_object"}
)

print(resp.choices[0].message.content)

{
  "cities": [
    {
      "name": "纽约",
      "description": "以其高楼大厦和繁忙的街道而闻名，是美国最大的都市之一。"
    },
    {
      "name": "伦敦",
      "description": "以其历史和文化而闻名，拥有丰富的博物馆、历史景点和文化活动。"
    },
    {
      "name": "巴黎",
      "description": "以其爱乐山和塞纳河而闻名，拥有迷人的历史、文化和美食。"
    },
    {
      "name": "里约热里奥",
      "description": "以其壮观的自然风景和热情洋溢的人民而闻名，是世界最热烈的天地之一。"
    },
    {
      "name": "悉尼",
      "description": "以其塔式高楼和环绕其周边的海滨气候而闻名，是澳大利亚最大的城市。"
    }
  ]
}

```

### 6. langchain的结构化输出代码示例
```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from icecream import ic

load_dotenv()

api_key =  "sk-b5e02d8f907b42f98044391e97f854ab"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
model="deepseek-v3"
# Configure an LLM
llm = ChatOpenAI(
    model=model,
    temperature=0.3,
    base_url=base_url,
    api_key=api_key,
    # streaming=True,
    # extra_body={"enable_thinking": False},
    # model_kwargs={
    #     "response_format": {"type": "json_object"}
    # }
)

# ic(llm.invoke("你好"))

# 方法1）
def 方法1_with_structured_output():
    from langchain_core.prompts import ChatPromptTemplate
    from pydantic import BaseModel, Field
    tagging_prompt = ChatPromptTemplate.from_template(
        """
        Extract the desired information from the following passage and return it in JSON format..
        passage:
        {input}"""
    )

    class Classification(BaseModel):
        sentiment: str = Field(description="The sentiment of the text")
        aggressiveness: int = Field(
            description="How aggressive the text is on a scale from 1 to 10"
        )
        language: str = Field(description="The language the text is written in")

    structured_llm = llm.with_structured_output(Classification)

    input = "I'm incredibly glad I met you! I think we'll be great friends!"
    prompt = tagging_prompt.invoke({"input": input})
    ic(prompt)
    response = structured_llm.invoke(prompt)
    print(response)

# 方法2) 
"""
方法1_with_structured_output 错误原因分享
 DeepSeek-V3（通过阿里 DashScope）虽然支持 response_format={"type": "json_object"}，它有时会返回 Markdown 包裹的 JSON，例如：
JSON
复制
```json
{
  "sentiment": "positive",
  "aggressiveness": 1,
  "language": "English"
}
复制
这就会导致 Pydantic 报错 `Invalid JSON: expected value at line 1 column 1`。
"""

from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
class Classification(BaseModel):
    sentiment: str = Field(description="The sentiment of the text")
    aggressiveness: int = Field(
        description="How aggressive the text is on a scale from 1 to 10"
    )
    language: str = Field(description="The language the text is written in")

parser = PydanticOutputParser(pydantic_object=Classification)

prompt = PromptTemplate(
    template="Extract the desired information from the following passage and return it in JSON format.\n{format_instructions}\n\nPassage:\n{input}",
    input_variables=["input"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | llm | parser

input_text = "I'm incredibly glad I met you! I think we'll be great friends!"
response = chain.invoke({"input": input_text})
print(response)

```
### 7.Outlines
解决思路：
  通过 prompt 来约束回答格式的方法事实上并不稳定。对于 json 来说，它总是以「{"」开始，并且每个 key 的右引号后必然是跟着一个 “:” ，因此如果能干预解码过程（采样空间），那将使得模型的回答格式更加可控。

Outlines 控制LLM按照指定格式输出：
1. 正则标的式  2.JSON Schema   3. 上下文无关语法-例如;SQL

项目地址：https://github.com/outlines-dev/outlines
官方博客： https://blog.dottxt.co/coalescence.html
代码示例：  https://outlines-dev.github.io/outlines/reference/generation/cfg/

结构化生成速度秘密是：正则表达式和有限状态机(FSM)之间的等价性。
为了理解它是如何工作的，我们需要将JSON正则表达式转换为FSM，生成的采样过程被状态变化的规则限制了。
[[Outlines.png|Open: file-20260309233508351.png]]
![[Outlines.png]]
- demo1）生成人名
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
"""Example of integrating `outlines` with `transformers`."""

from enum import Enum
from pydantic import BaseModel

class Name(str, Enum):
    john = "John"
    paul = "Paul"

class Age(int, Enum):
    twenty = 20
    thirty = 30

class Character(BaseModel):
    name: Name  # name的取值是枚举的
    age: Age # age的取值也是枚举的

from outlines import models, generate
model = models.transformers("/home/qyc/bert/Qwen2-0.5B")
generator = generate.json(model, Character)
char = generator("Generate a young character named Paul.")
print(char) # name=<Name.paul: 'Paul'> age=<Age.twenty: 20>
print(repr(char)) # Character(name:"Paul", age:20)
```

### 8.lm-format-enforcer库
[[lm-format-enforcer库.png|Open: file-20260309233245999.png]]
![[lm-format-enforcer库.png]]
项目地址：https://github.com/noamgat/lm-format-enforcer
```python
# https://github.com/noamgat/lm-format-enforcer
from transformers import pipeline
from datasets import load_dataset
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel

model_id = "/home/qyc/bert/Qwen2-0.5B-Instruct-bnb-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
print("加载完原版模型")
EOS_TOKEN = tokenizer.eos_token  # 必须添加 EOS_TOKEN 这个特殊符号，否则生成会无限循环。。
tokenizer.pad_token_id = 128001

pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    return_full_text=False,
)

input_text = """你是一名医生,请根据手术记录判断当前手术名称在手术过程中存在相关编码依据。\n        当前手术记录:\n###手术过程: 患者在手术室喉罩全麻下成功进镜，见左主支气管新生物完全阻塞管腔，于电圈套逐步切除左主支气管新生物，后见左上叶管腔完全通畅；新生物根部位于左下叶，电圈套逐步切除左下叶部分新生物，后左下叶背段及内前基底段管腔通畅，术中患者生命体征平稳，嘱术毕2小时后开始进食，注意呼吸情况及咯血量。###\n手术名称:###内镜下支气管病损切除术###，请判断手术名称是否合理并给出手术过程中出现的相关依据。"""
prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f'{input_text}'}
    ],
    tokenize=False,
    add_generation_prompt=True  # 添加模型开始预测的标签字符
)

from langchain_experimental.llms import LMFormatEnforcer
from pydantic import BaseModel
class AnswerFormat(BaseModel):
    手术名称: str
    手术是否存在依据: str
    相关依据: str


langchain_pipeline = LMFormatEnforcer(pipeline=pipe, json_schema=AnswerFormat.schema())

prompts = [prompt]

result = langchain_pipeline(prompts[0])

results = langchain_pipeline.generate(prompts)
for generation in results.generations:
    print(generation[0].text)
```
### 9. vllm支持结构化输出
```python
 python -m vllm.entrypoints.openai.api_server --model "Meta-Llama-3.1-8B-Instruct" --trust-remote-code --dtype float16 --tensor-parallel-size 2  --max-model-len 8096   --guided-decoding-backend=outlines

from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.0.172:8000/v1",
    api_key="EMPTY",
)

message = {"role": "user", "content": "Hello!"}
completion = client.chat.completions.create(
    model="Meta-Llama-3.1-8B-Instruct",
    messages=[
        message
    ],
    extra_body={
        "guided_choice": ["positive", "negative"]
    }
)
print(completion.choices[0].message.content)
```

### 10.KOR 使用LLM提取文本中的结构化数据
  项目地址： https://github.com/eyurtsev/kor/blob/main/docs
   - 示例1
```python
from kor import create_extraction_chain, Object, Text
from kor.nodes import Object, Text, Number
import os
from langchain.chat_models import ChatOpenAI

schema = Object(
    id="player",
    description=(
        "User is controlling a music player to select songs, pause or start them or play"
        " music by a particular artist."
    ),
    attributes=[
        Text(
            id="song",
            description="User wants to play this song",
            examples=[],
            many=True,
        ),
        Text(
            id="album",
            description="User wants to play this album",
            examples=[],
            many=True,
        ),
        Text(
            id="artist",
            description="Music by the given artist",
            examples=[("Songs by paul simon", "paul simon")],
            many=True,
        ),
        Text(
            id="action",
            description="Action to take one of: `play`, `stop`, `next`, `previous`.",
            examples=[
                ("Please stop the music", "stop"),
                ("play something", "play"),
                ("play a song", "play"),
                ("next song", "next"),
            ],
        ),
    ],
    many=False,
)
os.environ['VLLM_USE_MODELSCOPE'] = 'True'
llm = ChatOpenAI(
    openai_api_base="http://192.168.0.172:8000/v1",
    openai_api_key="EMPTY",
    model="Meta-Llama-3.1-8B-Instruct",
    stop="<|im_end|>",  # chatglm3-6b 用 <|im_end|>， llama3-6b 用 <|eot_id|>
    temperature=0
)
## chain
chain = create_extraction_chain(llm, schema, encoder_or_encoder_class='json')
print(chain.invoke("play songs by paul simon and led zeppelin and the doors"))
# >>>{'data': {'player': {'artist': ['paul simon', 'led zeppelin', 'the doors']}}, 'raw': '<json>{"player": {"artist": ["paul simon", "led zeppelin", "the doors"]}}</json>', 'errors': [], 'validated_data': {}}
```
- 示例2
```python
# KOR 结合 Pydantic Schema — Json 对象的输出列表
from kor import from_pydantic
from typing import List, Optional
from pydantic import BaseModel, Field


## schema
class PlanetSchema(BaseModel):
    planet_name: str = Field(description="The name of the planet")


class PlanetList(BaseModel):
    planets: List[PlanetSchema]


schema, validator = from_pydantic(
    PlanetSchema,
    description="Planet Information",
    many=True,  # <-- Note Many = True
)

chain = create_extraction_chain(llm, schema, validator=validator)

result = chain.invoke(("list planets in our solar system."))  # 列出我们太阳系中的行星。
print(result)
```

### 11.lm-format-enforcer库
项目地址: https://github.com/noamgat/lm-format-enforcer
该项目通过过滤语言模型在每个时间步允许生成的标记来解决问题，从而确保尊重输出格式，同时最大限度地减少语言模型的限制。
但是会导致推理时间变大几倍。
```python

from transformers import pipeline
from datasets import load_dataset
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel

model_id = "/home/qyc/bert/Qwen2-0.5B-Instruct-bnb-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
print("加载完原版模型")
EOS_TOKEN = tokenizer.eos_token  # 必须添加 EOS_TOKEN 这个特殊符号，否则生成会无限循环。。
tokenizer.pad_token_id = 128001

pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    return_full_text=False,
)

input_text = """你是一名医生,请根据手术记录判断当前手术名称在手术过程中存在相关编码依据。\n        当前手术记录:\n###手术过程: 患者在手术室喉罩全麻下成功进镜，见左主支气管新生物完全阻塞管腔，于电圈套逐步切除左主支气管新生物，后见左上叶管腔完全通畅；新生物根部位于左下叶，电圈套逐步切除左下叶部分新生物，后左下叶背段及内前基底段管腔通畅，术中患者生命体征平稳，嘱术毕2小时后开始进食，注意呼吸情况及咯血量。###\n手术名称:###内镜下支气管病损切除术###，请判断手术名称是否合理并给出手术过程中出现的相关依据。"""
prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f'{input_text}'}
    ],
    tokenize=False,
    add_generation_prompt=True  # 添加模型开始预测的标签字符
)

from langchain_experimental.llms import LMFormatEnforcer
from pydantic import BaseModel


class AnswerFormat(BaseModel):
    手术名称: str
    手术是否存在依据: str
    相关依据: str


langchain_pipeline = LMFormatEnforcer(pipeline=pipe, json_schema=AnswerFormat.schema())

prompts = [prompt]

result = langchain_pipeline(prompts[0])

results = langchain_pipeline.generate(prompts)
for generation in results.generations:
    print(generation[0].text)

```