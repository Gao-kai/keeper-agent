import logging


def setup_logging(level: int = logging.INFO):
    """
    配置全局基础日志系统。

    该函数通过调用 logging.basicConfig 为整个应用程序设置统一的日志格式、
    日期格式和最低记录级别。通常在程序的主入口（如 main.py 或 app.py）中调用一次。

    Args:
        level (int): 日志记录的最低级别。默认为 logging.INFO。
                     传入 logging.DEBUG 可开启更详细的调试日志，
                     传入 logging.WARNING 则只显示警告及以上级别的日志。

    Returns:
        None: 该函数无返回值，仅产生配置日志系统的全局副作用。

    Note:
        ⚠️ 关键警告：logging.basicConfig() 仅在根日志记录器（Root Logger）
        尚未配置任何处理器（Handler）时生效。
        如果该函数被多次调用，或者在调用前已经有其他模块（如第三方库）
        配置过日志，后续的 basicConfig 调用将被静默忽略，不会更新配置。
        如需强制重新配置，请使用 logging.basicConfig(..., force=True)（Python 3.8+）。

    Example:
        >>> setup_logging()  # 使用默认的 INFO 级别
        >>> setup_logging(logging.DEBUG)  # 开启 DEBUG 级别日志
    """

    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True,
        handlers=[logging.StreamHandler()] # 输出到控制台
    )
