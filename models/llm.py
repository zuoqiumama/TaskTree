import time
from typing import Optional, Union, Any
import sys
import os

# 将项目根目录添加到 sys.path，以便能找到 utils 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.silicon import SiliconFlow

class LLM:
    def __init__(
        self, 
        api_key: str, 
        system_prompt: str = None,
        model_name: str = "Qwen/Qwen3-8B",
        base_url: str = "https://api.siliconflow.cn/v1"
    ):
        """
        初始化 LLM 客户端
        
        Args:
            api_key: SiliconFlow API Key
            model_name: 模型名称
            base_url: API 基础 URL
        """
        self.client = SiliconFlow(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.system_prompt = system_prompt

    def query(
        self, 
        ins: str, 
        n: int = 1, 
        enable_thinking: bool = False,
        max_tokens: int = 2048,
        temperature: float = 0.5,
        max_retries: int = 10
    ) -> Union[str, Any, None]:
        """
        查询大模型，包含重试机制 (基于 query_llm 实现)
        
        Args:
            prompt: System Prompt (系统提示词)
            ins: User Instruction (用户指令/输入)
            n: 返回的候选项数量
            enable_thinking: 是否启用思考能力
            max_tokens: 最大生成长度
            temperature: 温度
            max_retries: 最大重试次数
            
        Returns:
            如果 n=1 返回内容字符串，否则返回完整 response 对象。失败返回 None。
        """
        try_times = 0
        response = None
        
        while True:
            try: 
                # 构建消息列表
                messages = [
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    },
                    {
                        "role": "user", 
                        "content": ins,
                    }
                ]

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    presence_penalty=0.5,
                    frequency_penalty=0.3,
                    n=n,
                    enable_thinking=enable_thinking,
                    thinking_budget=2048 if enable_thinking else None
                )
                # 成功获取响应，跳出循环
                break 
            except Exception as e:
                try_times += 1
                print(f"[尝试 {try_times}/{max_retries}] LLM查询失败: {type(e).__name__}: {str(e)}")
                time.sleep(5)
            
            if try_times >= max_retries:
                print(f"[警告!!!!!] LLM查询尝试次数达到{max_retries}次，最后一次错误: {type(e).__name__}: {str(e)}")
                return None

        # 循环结束后处理返回值
        if n == 1:
            return response.choices[0].message.content
        else:
            return response
        
# def test_llm_connection():
#     # 这里填入你的 API Key，或者从环境变量获取
#     # 注意：不要将真实的 Key 提交到版本控制系统中
#     api_key = "sk-xwucfzugonxtxwpuopkwufuverlbwvurulsvgwyrxqjrqjuq" 
    
#     if not api_key:
#         print("错误: 请在代码中设置 API Key")
#         return

#     print(f"正在初始化 LLM (Model: Qwen/Qwen3-8B)...")
#     llm = LLM(api_key=api_key)
    
#     system_prompt = "你是一个乐于助人的AI助手。"
#     user_instruction = "请用一句话介绍你自己，并告诉我今天是几号（假设今天是2025年11月27日）。"
    
#     print(f"\n发送请求:\nSystem: {system_prompt}\nUser: {user_instruction}\n")
#     print("正在等待回复...")
    
#     try:
#         # 测试普通查询
#         response = llm.query(
#             prompt=system_prompt,
#             ins=user_instruction,
#             enable_thinking=False
#         )
        
#         if response:
#             print("\n" + "="*20 + " 测试成功 " + "="*20)
#             print(f"模型回复:\n{response}")
#             print("="*50)
#         else:
#             print("\n" + "="*20 + " 测试失败 " + "="*20)
#             print("未收到回复 (返回 None)")
            
#     except Exception as e:
#         print(f"\n发生未捕获的异常: {e}")


# test_llm_connection()