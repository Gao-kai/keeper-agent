from pydantic import BaseModel, Field

"""
✅ Python Tips: [BaseModel]
BaseModel就是让所有基础类都具有类型校验和序列化能力的基类

UploadResponse 继承自 Pydantic 的 BaseModel，声明它是一个数据模型类（Schema）。
定义这样的类后，Pydantic 会自动提供：

1. 类型校验：类中声明的字段类型（如 int、str）会在赋值/反序列化时自动校验，类型不对会报错
2. 数据解析：可以直接从 JSON 字符串或 dict 构造实例，例如：
   response = UploadResponse.model_validate({"code": 0, "message": "ok"})
3. 序列化：model_dump() 转 dict，model_dump_json() 转 JSON
4. 自动生成文档：配合 FastAPI 时自动生成 OpenAPI 接口文档
"""
class UploadResponse(BaseModel):
	"""
	定义上传文件接口返回类型
	Filed中的...省略号表示构建UploadResponse实例时必须提供此字段，若不传递则校验报错
	"""
	message:str = Field(...,description="响应消息")
	task_id:str = Field(...,description="任务ID")
