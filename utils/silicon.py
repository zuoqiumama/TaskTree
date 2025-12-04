import requests
import json
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Message:
    role: str
    content: str
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None

@dataclass
class Choice:
    message: Message
    finish_reason: str
    index: int = 0

@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

@dataclass
class ChatCompletion:
    id: str
    choices: List[Choice]
    usage: Usage
    created: int
    model: str
    object: str = "chat.completion"

class ChatCompletions:
    def __init__(self, api_key: str, base_url: str = "https://api.siliconflow.cn/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        
    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        n: Optional[int] = None,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        response_format: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
        # SiliconFlow 特有参数
        enable_thinking: Optional[bool] = None,
        thinking_budget: Optional[int] = None,
        **kwargs
    ) -> ChatCompletion:
        """创建聊天完成请求"""
        
        url = f"{self.base_url}/chat/completions"
        
        # 构建请求负载
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        
        # 添加可选参数
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if top_k is not None:
            payload["top_k"] = top_k
        if min_p is not None:
            payload["min_p"] = min_p
        if n is not None:
            payload["n"] = n
        if stop is not None:
            payload["stop"] = stop
        if presence_penalty is not None:
            payload["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            payload["frequency_penalty"] = frequency_penalty
        if response_format is not None:
            payload["response_format"] = response_format
        if tools is not None:
            payload["tools"] = tools
            
        # SiliconFlow 特有参数
        if enable_thinking is not None:
            payload["enable_thinking"] = enable_thinking
        if thinking_budget is not None:
            payload["thinking_budget"] = thinking_budget
            
        # 添加其他kwargs参数
        payload.update(kwargs)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            # 解析响应
            data = response.json()
            return self._parse_response(data)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"API请求失败: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"响应解析失败: {str(e)}")
    
    def _parse_response(self, data: Dict) -> ChatCompletion:
        """解析API响应为ChatCompletion对象"""
        choices = []
        
        for i, choice_data in enumerate(data.get("choices", [])):
            message_data = choice_data.get("message", {})
            
            # 获取原始内容和思考内容
            original_content = message_data.get("content", "")
            reasoning_content = message_data.get("reasoning_content", "")
            
            # 合并思考内容和正式内容
            if reasoning_content:
                # 将思考内容放在正式内容前面，用<think>标签包围
                merged_content = f"<think>{reasoning_content}</think>{original_content}"
            else:
                merged_content = original_content
            
            message = Message(
                role=message_data.get("role", "assistant"),
                content=merged_content,  # 使用合并后的内容
                reasoning_content=reasoning_content,  # 保留原始思考内容（可选）
                tool_calls=message_data.get("tool_calls")
            )
            
            choice = Choice(
                message=message,
                finish_reason=choice_data.get("finish_reason", "stop"),
                index=i
            )
            choices.append(choice)
        
        usage_data = data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0)
        )
        
        return ChatCompletion(
            id=data.get("id", ""),
            choices=choices,
            usage=usage,
            created=data.get("created", int(datetime.now().timestamp())),
            model=data.get("model", ""),
            object=data.get("object", "chat.completion")
        )

class Chat:
    def __init__(self, api_key: str, base_url: str = "https://api.siliconflow.cn/v1"):
        self.completions = ChatCompletions(api_key, base_url)

class SiliconFlow:
    """SiliconFlow 客户端，模拟 OpenAI 接口"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.siliconflow.cn/v1"):
        """
        初始化SiliconFlow客户端
        
        Args:
            api_key: API密钥
            base_url: API基础URL
        """
        self.api_key = api_key
        self.base_url = base_url
        self.chat = Chat(api_key, base_url)


# client = SiliconFlow(
#     api_key="sk-xwucfzugonxtxwpuopkwufuverlbwvurulsvgwyrxqjrqjuq",
#     base_url="https://api.siliconflow.cn/v1"
# )


# response = client.chat.completions.create(
#     model="Qwen/Qwen3-8B",  # 使用支持思考的模型
#     messages=[
#         {"role": "system", "content": "你是一个有用的AI助手。"},
#         {"role": "user", "content": "请解释一下量子计算的基本原理，并思考它的应用前景。"}
#     ],
#     max_tokens=512,
#     temperature=0.7,
#     top_p=0.9,
#     n=1,
#     frequency_penalty=0.3,
#     # 启用思考功能
#     enable_thinking=True,
#     thinking_budget=2048,
#     min_p=0.05
# )
    
# # 打印响应
# print("\n===== SiliconFlow API响应（含思考内容）=====")
# print(f"状态: 成功")
# print(f"模型: {response.model}")
# print(f"完整内容（包含思考）:")
# print("-" * 50)
# print(response.choices[0].message.content)
# print("-" * 50)
# print(f"完成原因: {response.choices[0].finish_reason}")
# print(f"使用的tokens: {response.usage.total_tokens}")
# print("=" * 50)