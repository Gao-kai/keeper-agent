from modelscope.hub.snapshot_download import snapshot_download
from dotenv import load_dotenv
import os, logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
load_dotenv(override=True)


def download_model_from_modelscope(model_id: str, cache_dir: str):
	if not model_id:
		logger.error(f"未配置模型名称")
		return
	
	if not cache_dir:
		logger.info(f"未配置模型缓存路径，下载至默认目录: {os.getenv("MODELSCOPE_CACHE")}")
		cache_dir = os.getenv("MODELSCOPE_CACHE")
	
	try:
		model_dir = snapshot_download(model_id, cache_dir)
		print(f"模型下载完成，本地路径为：{model_dir}")
	except Exception as e:
		logger.error(f"从ModelScope平台下载模型{model_id}失败，请稍后重试: {e}")


if __name__ == "__main__":
	download_model_from_modelscope(model_id="BAAI/bge-m3", cache_dir='/Users/artest/.cache/modelscope')
