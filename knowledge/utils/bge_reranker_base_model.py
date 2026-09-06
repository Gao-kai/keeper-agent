import os
import logging
from typing import Optional

from dotenv import load_dotenv
from FlagEmbedding import FlagReranker

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
load_dotenv(override=True)

bge_reranker_model: Optional[FlagReranker] = None


def get_bge_reranker_model() -> Optional[FlagReranker]:
    """获取 BGE Reranker 重排序模型（单例模式）。

    从环境变量读取模型配置并初始化 FlagReranker 实例，
    首次调用时创建，后续调用直接复用已创建的实例，避免重复加载模型。

    环境变量:
        BGE_RERANKER_BASE_PATH: 本地模型路径，默认 "BAAI/bge-reranker-base"
            （取不到时会自动尝试从 HuggingFace 下载）
        BGE_RERANKER_DEVICE: 运行设备，如 'cpu' / 'mps' / 'cuda:0'，默认 "cpu"
        BGE_RERANKER_FP16: 是否启用半精度，默认 False
        BGE_RERANKER_NORMALIZE: 是否用 sigmoid 把分数归一化到 [0, 1]，默认 True

    Returns:
        Optional[FlagReranker]: 初始化好的重排序模型实例；失败返回 None。
    """
    global bge_reranker_model

    try:
        if bge_reranker_model is not None and isinstance(
                bge_reranker_model, FlagReranker
        ):
            return bge_reranker_model

        # 防止字符串中斜杠等符号被转义
        model_name_or_path = os.path.normpath(
            os.getenv("BGE_RERANKER_BASE_PATH", "BAAI/bge-reranker-base")
        )
        # 新版 FlagEmbedding 的参数名是 devices（复数），传 device 会被 kwargs 吞掉而不生效
        model_device = os.getenv("BGE_RERANKER_DEVICE", "cpu")

        # os env读取的值是字符串 需要转化为模型需要参数布尔值
        use_fp16 = os.getenv("BGE_RERANKER_FP16", "false").lower() in (
            "true", "1", "yes"
        )
        # 归一化后分数落在 [0,1]，与断崖检测阈值（gap_abs/gap_ratio）量纲匹配；
        # 不归一化时输出的是 logits（如 7.37 / -10.19），阈值会完全失效
        normalize = os.getenv("BGE_RERANKER_NORMALIZE", "true").lower() in (
            "true", "1", "yes"
        )

        bge_reranker_model = FlagReranker(
            model_name_or_path=model_name_or_path,
            devices=model_device,
            use_fp16=use_fp16,
            normalize=normalize,
        )

        logger.info(
            f"BGE Reranker 加载完成: path={model_name_or_path}, "
            f"devices={bge_reranker_model.target_devices}, normalize={normalize}"
        )
        return bge_reranker_model
    except Exception as e:
        # 用 exception 打印完整堆栈，避免只看到 str(e) 无法定位问题
        logger.exception(f"初始化bge_reranker_model重排序模型失败: {e}")
        return None


if __name__ == "__main__":
    model = get_bge_reranker_model()
    if model:
        print(model.compute_score([
            ("什么是万用表？", "万用表是一种测量电压、电流、电阻的仪器"),
            ("什么是万用表？", "今天天气很好"),
        ]))
