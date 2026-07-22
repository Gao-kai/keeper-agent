"""
导入功能 - 节点基类定义模块

# 功能
    1. 基于类的继承和抽象类定义统一的节点接口
    2. 提供日志打印、任务追踪、异常处理通用接口

# 核心
    1. 节点开始结束以及中间各环节日志记录
    2. 注册当前执行的节点，提供任务追踪
    3. 统一包装异常，添加节点信息
    4. 配置注入和默认配置

# 流程
    1. 子类继承BaseNode基类
    2. 实例化子节点类得到实例
    3. 执行调用实例会执行父类的__call__方法，从而执行实例自己的process方法
    4. 子类控制process方法实现即可完全控制节点的行为
"""
import logging
from abc import ABC, abstractmethod
from typing import Optional, TypeVar

from knowledge.processor.import_process.config import ImportConfig, get_import_config
from knowledge.processor.import_process.exception import ImportProcessError
from knowledge.processor.import_process.state import ImportGraphState

T = TypeVar("T")  # 泛型状态类型


class BaseNode(ABC):
    # 节点名称 子类应该实现重写
    name: str = "base_node"

    def __init__(self, config: Optional[ImportConfig] = None):
        """
        初始化节点
        Args:
            config: 配置对象，不传递默认使用全局配置
        """
        self.config = config or get_import_config()
        # 便于按照节点过滤日志
        self.logger = logging.getLogger(f"import.{self.name}")

    def __call__(self, state: T) -> T:
        """
        节点执行入口

        LangGraph 调用节点时会调用此方法，为系统提供统一的日志输出、任务追踪和异常处理。
        调用此方法会进一步调用每一个节点类上的process方法

        Args:
            state: 图状态字典

        Returns:
            state: 更新后的图状态字典

        Raises:
            ImportProcessError 节点执行失败时抛出异常
        """
        try:
            result = self.process(state)
            self.logger.info(f"--- {self.name} 节点执行完成 ---")
            return result
        # 异常已经被包装为ImportProcessError 直接放行 交给上层调用者
        except ImportProcessError:
            raise
        # 异常为其他异常未包装 包装成自定义异常ImportProcessError后抛出
        except Exception as e:
            self.logger.error(f"--- {self.name} 节点执行失败 ---")
            raise ImportProcessError(
                message=str(e),
                node_name=self.name,
                cause=e
            )

    @abstractmethod
    def process(self, state: T) -> T:
        """
        节点核心处理逻辑
        所有子类必需实现此抽象方法
        Args:
            state:  图状态字典

        Returns:
            state: 更新后的图状态字典

        """
        pass

    def log_step(self, step_name: str, message: str = ''):
        """
        步骤日志记录
        Args:
            step_name: 步骤名称
            message: 附加信息

        Returns:

        """
        self.logger.info(f"[{step_name}]{message}")
