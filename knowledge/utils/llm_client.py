from logging import INFO
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os,logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=INFO)
load_dotenv(override=True)

llm_model_cache = {}

def get_llm_client(
		model: str = '',
		temperature: float = 0.1,
		response_json: bool = False
):
	
	try:
		# 获取模型参数
		model_name = model or os.getenv("LLM_DEFAULT_MODEL")
		api_key = os.getenv("OPENAI_API_KEY")
		base_url = os.getenv("OPENAI_API_BASE")
		
		# 利用元组不可变的特性 生成缓存Key
		cache_key = (model_name, temperature, response_json)
		if llm_model_cache.get(cache_key):
			return llm_model_cache.get(cache_key)
		
		# 校验
		if not model_name:
			raise ValueError(f"LLM模型初始化失败，模型名称model_name为空")
		
		if not api_key:
			raise ValueError(f"LLM模型初始化失败，APIKEY为空")
		
		if not base_url:
			raise ValueError(f"LLM模型初始化失败，base_url为空")
		
		# 初始化LLM客户端
		model_kwargs = {}
		if response_json:
			model_kwargs["response_format"] = {
				"type": "json_object"
			}
		
		# kimi-k3 模型的 temperature 固定为 1.0 不可修改，显式传入其他值会报
		# invalid_request_error；传 None 让 langchain 序列化请求时跳过该字段，
		# 由服务端使用默认值
		if "kimi" in model_name.lower():
			temperature = None
		
		llm_client = ChatOpenAI(
			model=model_name,
			temperature=temperature,
			base_url=base_url,
			api_key=api_key,
			extra_body={
				"enable_thinking": False
			},
			model_kwargs=model_kwargs
		)
		
		llm_model_cache[cache_key] = llm_client
		
		return llm_client
	except Exception as e:
		logger.error(f"创建LLM客户端报错: {e}")
		return None


if __name__ == "__main":
	llm = get_llm_client()
