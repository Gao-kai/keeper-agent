import os

import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from knowledge.api.import_route import create_import_api_route
from knowledge.api.query_router import create_query_api_route
from knowledge.utils.path_utils import get_static_dir_name


def create_fastapi_app():
    # 创建FastAPI实例
    app = FastAPI(
        title="RAG知识库 API服务",
        summary="Slow is Fast.",
        description="Provide RAG Knowledge Base API via browser",
    )

    # 跨域中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 运行任意来源的URL请求当前服务
        allow_methods=["*"],  # 任意请求方法
        allow_headers=["*"],  # 任意自定义的请求头
        allow_credentials=True,
    )

    # 注册API请求路由
    create_import_api_route(app)
    create_query_api_route(app)

    # 静态资源挂载（必须放在API路由之后，根路径兜底处理前端构建产物）
    front_static_dir_name = get_static_dir_name()

    if front_static_dir_name and os.path.exists(front_static_dir_name):
        # 兼容旧路径：/static/* 前缀
        # mount("/static") 只在 /static/xxx 处开门
        app.mount("/static", StaticFiles(directory=front_static_dir_name))

        # mount("/") 则在所有未匹配路径处开门兜底
        # 根路径挂载：index.html 中引用的 /vite.svg、/assets/* 等资源可直接访问；
        # html=True 使访问 / 时自动返回 index.html
        app.mount("/", StaticFiles(directory=front_static_dir_name, html=True))

    # 返回FastAPI实例
    return app


if __name__ == "__main__":
    uvicorn.run(app=create_fastapi_app(), host="127.0.0.1", port=8000)
