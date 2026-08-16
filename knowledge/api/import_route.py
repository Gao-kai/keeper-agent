import os.path
import uvicorn
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

from knowledge.schema.task_schema import TaskStatusResponse
from knowledge.schema.upload_schema import UploadResponse
from knowledge.utils.project_path import get_front_static_dir_name


def create_fastapi_app():
	# 创建FastAPI实例
	app = FastAPI(
		title="RAG知识库导入文件API服务",
		summary="Slow is Fast.",
		description="Provide upload file API via browser"
	)
	
	# 跨域中间件
	app.add_middleware(
		CORSMiddleware,
		allow_origins= ["*"], # 运行任意来源的URL请求当前服务
		allow_methods = ["*"], # 任意请求方法
		allow_headers =["*"], # 任意自定义的请求头
		allow_credentials=True,
		)
	
	# 静态资源中间件
	front_static_dir_name = get_front_static_dir_name()
	if front_static_dir_name and os.path.exists(front_static_dir_name):
		app.mount("/static",StaticFiles(directory=front_static_dir_name))
	

	# API请求路由
	create_app_route(app)
	
	# 返回FastAPI实例
	return app
	
def create_app_route(app):
	"""
	1. 普通API请求
	2. 静态资源请求
	3. 前端构建打包资源如何挂载对静态资源服务器上
	4. 如何配置Nginx实例（真实打包挂载）
	"""
	
	# 入口页面静态资源请求
	@app.get('/index')
	async def get_hello():
		front_static_dir_name = get_front_static_dir_name()
		return FileResponse(
			path=os.path.join(front_static_dir_name,'index.html')
		)
	
	# 上传接口
	@app.post("/upload_file/",response_model=UploadResponse)
	async def upload_file(file: UploadFile):
		# 1. 将用户上传的文件上传至Minio服务
		# task_id, import_file_path, file_dir =
		# 2. 开始节点编排流程
		
		# 3. 返回响应结果
		pass
	
	# 基于task_id查询当前任务状态
	@app.get("/status/{task_id}",response_model=TaskStatusResponse)
	async def get_task_status(task_id:str):
		pass
		
	
	

if __name__ == "__main__":
	uvicorn.run(
		app=create_fastapi_app(),
		host="127.0.0.1",
		port=8000
	)