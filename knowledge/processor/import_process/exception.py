"""
导入功能 - 异常捕获模块

# 功能
    1. 基于Python中类的继承实现层级清晰的异常分类
    2. 每一个异常都携带上下文信息，比如当前异常节点名、异常原因
    3. 支持异常链追溯，提供更加清晰的异常信息

# 流程
    1. 程序异常发生
    2. 捕获原始异常cause
    3. 包装为自定义异常
        - 添加节点名称node_name
        - 保留原始异常caus
        - 自定义友好错误消息格式
    4. 包装后向上游抛出，由调用方进行处理
    
# 分类
    ImportProcessError (基础异常)
    ├── ConfigurationError      # 配置错误
    ├── FileProcessingError     # 文件处理错误
    │   ├── PdfConversionError  # PDF 转换错误
    │   └── ImageProcessingError# 图片处理错误
    ├── DocumentSplitError      # 文档切分错误
    ├── EmbeddingError          # 向量化错误
    ├── LLMError                # LLM 调用错误
    ├── StorageError            # 存储错误
    │   ├── MilvusError         # Milvus 存储错误
    │   ├── Neo4jError          # Neo4j 存储错误
    │   └── MinioError          # MinIO 存储错误
    └── ValidationError         # 数据验证错误

"""


class ImportProcessError(Exception):
    def __init__(self, message: str, node_name: str, cause: Exception = None):
        """
        初始化异常
        Args:
            message: 自定义错误消息
            node_name: 节点名称
            cause: 原始异常
        """
        self.node_name = node_name
        self.cause = cause
        super().__init__(message)  # TODO 注意不要忘记初始化父类Exception

    def __str__(self):
        """
        格式化错误信息 直接打印异常对象时会调用此魔术方法
        Returns:

        """
        error_msg_list = []
        if self.node_name:
            error_msg_list.append(f"[节点名称：{self.node_name}]")
        error_msg_list.append(f"异常描述：{super().__str__()}")
        if self.cause:
            error_msg_list.append(f"(异常原因: {self.cause})")
        return " ".join(error_msg_list)


class ConfigurationError(ImportProcessError):
    """配置错误：环境变量缺失或配置值无效"""
    pass


class FileProcessingError(ImportProcessError):
    """文件处理错误：文件不存在、格式错误、读写失败"""
    pass


class PdfConversionError(FileProcessingError):
    """PDF 转换错误：MinerU 转换失败"""
    pass


class ImageProcessingError(FileProcessingError):
    """图片处理错误：图片总结、上传失败"""
    pass


class DocumentSplitError(ImportProcessError):
    """文档切分错误：切分逻辑异常"""
    pass


class EmbeddingError(ImportProcessError):
    """向量化错误：模型调用失败、向量生成异常"""
    pass


class LLMError(ImportProcessError):
    """LLM 调用错误：API 调用失败、响应解析失败"""
    pass


class StorageError(ImportProcessError):
    """存储错误：数据库操作失败"""
    pass


class MilvusError(StorageError):
    """Milvus 存储错误"""
    pass


class Neo4jError(StorageError):
    """Neo4j 存储错误"""
    pass


class MinioError(StorageError):
    """MinIO 存储错误"""
    pass


class ValidationError(ImportProcessError):
    """数据验证错误：输入数据不符合预期"""
    pass


if __name__ == "__main__":
    try:
        raise TypeError("代码异常")
    except Exception as e:
        raise PdfConversionError(
            message="PDF转换失败",
            node_name="PDF NODE",
            cause = e
        )
