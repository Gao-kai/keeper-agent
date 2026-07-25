"""
导入功能 - 状态定义模块

# 功能
    1. 定义图Graph状态的完整结构
    2. 提供默认工厂函数
    3. 文档化每个字段的用途


# 分类
    - 任务标识	task_id	任务追踪 ID
    - 控制标志	is_pdf_read_enabled, is_md_read_enabled	文件类型标志
    - 路径信息	import_file_path, file_dir, pdf_path, md_path	文件路径
    - 文件信息	file_title, item_name	元数据
    - 处理数据	md_content, chunks	中间结果

"""
import copy
from typing import TypedDict, List, Dict

"""
TypedDict: TypedDict 允许为每个键指定明确类型,提供静态检查
total=False: 表示字典中所有字段都为可选字段
"""
class ImportGraphState(TypedDict, total=False):
    # ==================== 任务标识 ====================
    task_id: str  # 任务 ID，用于任务追踪

    # ==================== 控制标志 ====================
    is_md_read_enabled: bool  # 是否启用 MD 读取
    is_pdf_read_enabled: bool  # 是否启用 PDF 读取

    # ==================== 路径信息 ====================
    import_file_path: str  # 导入文件路径（原始输入）
    file_dir: str  # 导入(出)文件目录
    pdf_path: str  # PDF 文件路径
    md_path: str  # 转换后 Markdown 文件路径

    # ==================== 文件信息 ====================
    file_title: str  # 文件标题（不含扩展名）
    item_name: str  # 识别出的商品/产品名称

    # ==================== 处理中间数据 ====================
    md_content: str  # Markdown 文档内容
    chunks: List  # 文档切片列表


# 默认状态模版
IMPORT_GRAPH_STATE_DEFAULT: ImportGraphState = {
    "task_id": "",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "file_dir": "",
    "import_file_path": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "md_content": "",
    "chunks": [],
    "item_name": "",
}

def create_default_state(**override_state: Dict) -> ImportGraphState:
    """
    工厂函数方式创建图状态实例
    Args:
        **override_state: 覆盖默认配置的字段组成的字典

    Returns:
        新的图状态实例
    """
    default_state = copy.deepcopy(IMPORT_GRAPH_STATE_DEFAULT)
    default_state.update(override_state)
    return default_state


def get_default_state()->ImportGraphState:
    """
    获取默图状态副本
    Returns:
        状态副本
    """
    return copy.deepcopy(IMPORT_GRAPH_STATE_DEFAULT)