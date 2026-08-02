"""
导入功能 - 配置管理模块

# 模块功能
    1. ✅ 集中管理所有配置项
    2. ✅ 支持环境变量覆盖
    3. ✅ 提供配置验证
    4. ✅实现全局单例模式

# 配置分类
    1. 文档基本配置：切片长度、拓展名
    2. LLM大模型配置
    3. 数据库配置
    4. 向量维度配置

# 配置加载流程
    1. load dotenv读取.env文件到os环境
    2. os getenv读取环境变量
    3. dataclass字段初始化 # TODO
    4. get_config返回全局单例

# 技术问题
    1. dataclass装饰类的效果
    2. 定义可变对象和不可变对象的区别
    3. 列表推导式 [表达式 for 变量 in 可迭代对象 if 条件]
    4. lamba表达式延迟加载 创建类不会求值 只有实例化时候才去求值
"""

import os
from dataclasses import field, dataclass
from typing import Set, Optional, List

from dotenv import load_dotenv

# 加载 .env文件 强制.env加载的环境变量优先级高于系统环境变量
load_dotenv(override=True)


@dataclass
class ImportConfig:
    """ 导入流程配置"""

    # ============== 文档处理配置 =============
    max_content_length: int = 2000  # 切片最大长度
    min_content_length: int = 500  # 合并短内容的最小长度
    max_image_context_length:int = 200 # 截取图片上下文最大长度
    overlap_sentences: int = 1  # 句子级切分时重叠句数
    item_name_chunk_k: int = 3  # 商品名识别时使用的切片数量

    # 支持的图片扩展名 TODO
    image_extensions: Set[str] = field(
        default_factory=lambda: {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    )

    # ==================== LLM 配置 ====================
    openai_api_base: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_BASE", "")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    vl_model: str = field(
        default_factory=lambda: os.getenv("VL_MODEL", "")
    )
    item_model: str = field(
        default_factory=lambda: os.getenv("ITEM_MODEL", "")
    )
    default_model: str = field(
        default_factory=lambda: os.getenv("MODEL", "")
    )

    # ==================== Milvus 配置 ====================
    milvus_url: str = field(
        default_factory=lambda: os.getenv("MILVUS_URL", "")
    )
    chunks_collection: str = field(
        default_factory=lambda: os.getenv("CHUNKS_COLLECTION", "")
    )
    item_name_collection: str = field(
        default_factory=lambda: os.getenv("ITEM_NAME_COLLECTION", "")
    )
    entity_name_collection: str = field(
        default_factory=lambda: os.getenv("ENTITY_NAME_COLLECTION", "")
    )

    # ==================== Neo4j 配置 ====================
    neo4j_uri: str = field(
        default_factory=lambda: os.getenv("NEO4J_URI", "")
    )
    neo4j_username: str = field(
        default_factory=lambda: os.getenv("NEO4J_USERNAME", "")
    )
    neo4j_password: str = field(
        default_factory=lambda: os.getenv("NEO4J_PASSWORD", "")
    )
    neo4j_database: str = field(
        default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j")
    )

    # ==================== MinIO 配置 ====================
    minio_endpoint: str = field(
        default_factory=lambda: os.getenv("MINIO_ENDPOINT", "")
    )
    minio_access_key: str = field(
        default_factory=lambda: os.getenv("MINIO_ACCESS_KEY", "")
    )
    minio_secret_key: str = field(
        default_factory=lambda: os.getenv("MINIO_SECRET_KEY", "")
    )
    minio_bucket: str = field(
        default_factory=lambda: os.getenv("MINIO_BUCKET_NAME", "")
    )
    minio_secure: bool = False

    # ==================== 向量配置 ====================
    embedding_dim: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1024"))
    )
    embedding_batch_size: int = 3

    # ==================== 速率限制 ====================
    requests_per_minute: int = 12  # 图片总结 API 速率限制

    @classmethod
    def from_env(cls) -> "ImportConfig":
        return cls()

    def validate(self, strict: bool = False) -> None:
        """
        校验配置是否完整
        Args:
            strict: 是否为严格模式，该模式下缺少配置会抛出异常

        Returns:

        """
        required_fields: List[str] = [
            "milvus_url",
            "chunks_collection"
        ]

        missing_fields = [key for key in required_fields if not getattr(self, key)]

        if missing_fields:
            msg = f"缺少必需配置: {missing_fields}"
            if strict:
                raise ValueError(msg)
            else:
                print(f"⚠️警告: {msg}")

    def get_minio_base_url(self) -> str:
        """
        获取Minio对象存储基础URL
        Returns:

        """
        protocol = "https" if self.minio_secure else "http"
        return f"{protocol}://{self.minio_endpoint}/{self.minio_bucket}"


# 定义全局配置实例对象
_import_config: Optional[ImportConfig] = None


def get_import_config() -> ImportConfig:
    """
    获取全局单例导入流程配置
    Returns:
        ImportConfig 导入流程配置
    """
    global _import_config
    if not _import_config:
        _import_config = ImportConfig.from_env()
    return _import_config
