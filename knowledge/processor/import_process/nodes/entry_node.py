"""
入口节点模块

# 功能
    1. 文件类型检测
    2. 路由导航

# 实现
    1. 文件类型检测与条件路由
    2. 使用MinerU进行PDF解析
    3. 使用subprocess子进程模块调用外部命令行工具
    4. 异常处理
    5. 单元测试
"""
import json

from knowledge.processor.import_process.base import BaseNode, T
from knowledge.processor.import_process.exception import ValidationError
from pathlib import Path
from knowledge.processor.import_process.log_config import setup_logging


class EntryNode(BaseNode):
    """
    入口节点

    基于state中文件的拓展名来处理不同的路由逻辑
    PDF文件 走PDF处理节点
    MC文件  走MD处理节点
    """
    name = "entry_node"

    def process(self, state: T) -> T:
        """
        入口节点核心处理逻辑
        Args:
            state: 图状态

        Returns:
            state: 更新后的图状态

        """
        self.log_step(step_name="STEP 01", message="获取文件路径")
        import_file_path = state.get("import_file_path", "")

        if not import_file_path:
            raise ValidationError(
                node_name=self.name,
                message="import_file_path不能为空",
            )

        file_path = Path(import_file_path)
        file_extension = file_path.suffix.lower()
        file_name = file_path.stem
        state["file_title"] = file_name
        self.log_step(step_name="STEP 02", message=f"检测到文件名:{file_name}")
        self.log_step(step_name="STEP 02", message=f"检测到文件类型:{file_extension}")

        if file_extension == '.pdf':
            self.logger.info("启用 PDF 读取流程")
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = import_file_path
        elif file_extension == '.md':
            self.logger.info("启用 MarkDown 读取流程")
            state["is_md_read_enabled"] = True
            state["md_path"] = import_file_path
        else:
            self.logger.warning(f"当前文件类型不支持向量化:{file_extension}")

        return state


if __name__ == '__main__':
    setup_logging()
    entry_node = EntryNode()
    process_state = entry_node({
        "import_file_path": "/Users/artest/Desktop/shopkeeper/docs/03_掌柜智库项目（导入处理）骨架代码.md"
    })
    print(json.dumps(process_state,indent=4,ensure_ascii=False))
