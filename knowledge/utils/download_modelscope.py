# from modelscope.hub.snapshot_download import snapshot_download
from modelscope import snapshot_download
from dotenv import load_dotenv
import os, logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
load_dotenv(override=True)


def download_model_from_modelscope(
        model_id: str,
        cache_dir: str = None,
        revision: str = "master",
):
    """从 ModelScope 平台下载模型到本地缓存目录。

    注意：新版 modelscope 的 snapshot_download 签名为
    (model_id, revision, cache_dir, ...)，第二个位置参数是 revision。
    因此这里必须全部用关键字参数传值，否则 cache_dir 会被当成 revision，
    导致平台返回空文件列表（'NoneType' object is not iterable）。

    Args:
        model_id: ModelScope 模型 ID，如 "BAAI/bge-m3"
        cache_dir: 模型缓存根目录，为空时读取环境变量 MODELSCOPE_CACHE
        revision: 模型版本分支，默认 "master"

    Returns:
        str: 模型下载完成后的本地快照路径；失败时返回 None
    """
    if not model_id:
        logger.error(f"未配置模型名称")
        return None

    if not cache_dir:
        cache_dir = os.getenv("MODELSCOPE_CACHE")
        logger.info(f"未配置模型缓存路径，下载至默认目录: {cache_dir}")

    try:
        model_dir = snapshot_download(
            model_id=model_id,
            cache_dir=cache_dir,
            revision=revision,
        )
        print(f"模型下载完成，本地路径为：{model_dir}")
        return model_dir
    except Exception as e:
        # 用 exception 打印完整堆栈，避免只看到 str(e) 无法定位问题
        logger.exception(f"从ModelScope平台下载模型{model_id}失败，请稍后重试: {e}")
        return None


if __name__ == "__main__":
    # 下载BAAI/bge-m3
    # download_model_from_modelscope(
    #     model_id="BAAI/bge-m3", cache_dir="/Users/artest/.cache/modelscope"
    # )

    # 下载BAAI/bge-reranker-base
    download_model_from_modelscope(
        model_id="BAAI/bge-reranker-base", cache_dir="/Users/artest/.cache/modelscope"
    )
