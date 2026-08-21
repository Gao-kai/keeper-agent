"""
PDF转化为MD节点模块

# 功能
    1. 将上一节点的PDF文件解析并转化为MD文件

# Mineru文档解析工具使用指南
    1. 功能: 将 PDF、图片以及 DOCX、PPTX、XLSX 转化为机器可读格式（如 Markdown、JSON），便于后续检索、抽取与二次处理
    2. 使用步骤
        - 安装python包 mineru[all] 否则无法解析Mineru的命令行参数
        - 切换模型源为modelscope export MINERU_MODEL_SOURCE=modelscope
        - 方法1: 执行命令行参数进行解析 mineru -p <input_path> -o <output_path>第一次会默认从modelscope远程下载模型后到本地的.cache目录，后续基于缓存
        - 方法2: 先本地下载所有模型 执行命令 mineru-models-download 会交互式选择modelscope或者huggingface
               设置后续解析过程模型源为本地 export MINERU_MODEL_SOURCE=local
        - Mineru在执行解析任务时会基于本地模型读取，模型存放位置需要在mineru.json配置文件中写入

# Mineru.json 配置文件
mineru -p /Users/artest/Desktop/shopkeeper/data/doc/迅饶网关与小米产品通讯配置说明.pdf -o /Users/artest/Desktop/shopkeeper/output

# Mineru解析产物分析

# Mineru本地解析流程

# Mineru 调用方式
    - python代码中基于api client进行调用 封装一个方法
    - python开子进程去执行命令行调用 基于subprocess


# TODO 需要补充下异常调用和打印的流程图
1. Python中任何未被捕获的异常都会导致程序直接终止，会先输出完整的异常调用链信息，然后底层调用错误对象e的__str__魔术方法来输出错误信息
2. 如果raise的自定义异常比如ValueError被except显式捕获后，此时捕获到的错误对象e直接打印就会调用__str__魔术方法来输出错误信息,但不会导致程序中止并且没有异常调用链
3. Python子进程的基础知识
4. Python客户端代码调用Mineru服务的Demo案例
"""
import os
import subprocess
import time
import json
from pathlib import Path
from typing import Tuple

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.exception import FileProcessingError, PdfConversionError
from knowledge.utils.log_config import setup_logging
from knowledge.processor.import_process.state import ImportGraphState


class PDFToMarkDownNode(BaseNode):
    name = "pdf_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        执行PDF转化为MD任务
        1. 验证逻辑
        2. 执行逻辑
        3. 路径计算
        4. 实时子进程日志输出
        5. 错误处理

        Args:
            state: 图状态

        Returns:
            state: 图状态

        """
        print(" PDF转化MarkDown 节点")

        # 1. 对输入路径进行校验
        self.log_step(step_name="STEP 01", message="校验PDF路径和解析产物输出路径")
        pdf_path_obj, output_dir_obj = self._validate_import_file_paths(state)

        # 2. 执行mineru转换 子进程方式
        self.log_step(step_name="STEP 02", message="执行PDF转MD文件核心任务")
        return_code = self._execute_mineru(state, pdf_path_obj, output_dir_obj)
        if return_code != 0:
            raise PdfConversionError(node_name=self.name, message="Mineru PDF转化MD失败")

        # 3. 获取md文件路径 更新state对象
        self.log_step(step_name="STEP 03", message="将转化后的MD文件路径写入图状态")
        state['md_path'] = self._get_md_path(pdf_path_obj,output_dir_obj)
        return state

    def _validate_import_file_paths(self, state: ImportGraphState) -> Tuple[Path,Path]:
        """
        验证 PDF路径和输出目录
        Args:
            state: 图状态

        Returns:
            (pdf_path_obj,output_dir_obj) 元组
        """
        # 1. 读取图状态中上一节点处理的PDF文件路径及为空校验
        pdf_path = state.get("pdf_path", "")
        if not pdf_path:
            raise FileProcessingError(node_name=self.name, message="PDF文件路径为空")

        # 2. 对输入路径标准化为Path对象及路径是否为真实文件系统路径合法校验
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise FileProcessingError(node_name=self.name, message="PDF文件路径不存在")

        # 3. 获取输出目录 如果为空则默认指定为当前解析PDF文件的文件夹目录
        output_dir = state.get("file_dir", "")
        if not output_dir:
            output_dir = str(pdf_path_obj.parent)
        output_dir_obj = Path(output_dir)

        self.logger.info(f"PDF路径：{pdf_path_obj.name}")

        # 4. 返回处理后的PDF路径和输出目录的Path对象
        return pdf_path_obj, output_dir_obj

    def _execute_mineru(self, state: ImportGraphState, pdf_path_obj: Path, output_dir_obj: Path) -> int:
        """
        已将Mineru Pipeline模型和VLM模型全部下载至本地.cache目录中并且mineru.json配置完成
        基于Python内置subprocess子进程调用命令行
        Args:
            state: 图状态
            pdf_path_obj: PDF文件路径对象
            output_dir_obj: 输出目录文件路径对象

        Returns:

        """
        # 1. 构建命令行命令
        command = [
            "mineru",
            "-p",
            str(pdf_path_obj),
            "-o",
            str(output_dir_obj),
            "--source",
            "local"
        ]
        self.logger.info(f"构建命令成功:{" ".join(command)}")
        start_time = time.time()

        # 2. 子进程调用命令行工具
        process = subprocess.Popen(
            args=command,
            stdout=subprocess.PIPE, # 将子进程的标准输出（stdout）重定向到管道，使父进程可通过 process.stdout 读取
            stderr=subprocess.STDOUT, # 将标准错误（stderr）合并到标准输出（stdout），统一通过 process.stdout 读取
            text=True,  # 以文本模式（而非字节模式）返回输出，自动将子进程的字节流解码为 Python 字符串
            encoding="utf-8",
            errors="replace", # 解码失败时用 � 替换非法字符，避免因个别乱码字符导致整个程序崩溃
            env=os.environ.copy(), # 复制当前环境变量传递给子进程，确保子进程能访问父进程的 PATH、模型路径等配置
            bufsize=1 # 设置行缓冲模式（1 表示行缓冲，0 为无缓冲，>1 为指定字节缓冲区）每行输出立即刷新
        )
        self.logger.info(f"本地开始执行Mineru解析命令")

        # 3. 逐行读取输出（非阻塞）
        for line in process.stdout:
            self.logger.info(f"[Mineru工具执行转换 {line.rstrip()}]")

        # 4. 等待子进程调用结束
        return_code = process.wait()
        end_time = time.time()

        # 5. 基于return_code判断命令执行结果
        if return_code == 0:
            self.logger.info(f"本地调用Mineru命令完成，总计耗时: {end_time - start_time:.2f} 秒")
        else:
            self.logger.info(f"本地调用Mineru命令失败： {return_code}")
        return return_code

    def _get_md_path(self, pdf_path_obj:Path, output_dir_obj:Path)->str:
        """
        基于输出目录Path对象和输入文件路径对象信息拼接输出MD文件路径信息
        path.name 文件名.后缀
        path.stem 文件名
        path.suffix 拓展名

        output_dir/
          └── 文件名/
               └── hybrid_auto/
                    ├── 文件名.md
                    └── images/

        Args:
            pdf_path_obj: pdf输入文件路径
            output_dir_obj: 输出目录文件路径

        Returns:
            md_path 解析后md文件存放路径
        """

        file_name = pdf_path_obj.stem
        md_path = output_dir_obj / file_name / "hybrid_auto" / f"{file_name}.md"
        return str(md_path)


if __name__ == "__main__":
    setup_logging()
    pdf_to_md_node = PDFToMarkDownNode()
    import_file_path = "/Users/artest/Desktop/shopkeeper/data/doc/万用表RS-12的使用.pdf"
    process_state = pdf_to_md_node({
        "pdf_path": import_file_path,
        "file_dir": "/Users/artest/Desktop/shopkeeper/output"

    })
    print(json.dumps(process_state, indent=4, ensure_ascii=False))