"""
创建Minio客户端
"""
import logging
import os
from minio import Minio
from dotenv import load_dotenv

load_dotenv(override=True)

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "111.228.53.183:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "knowledge-base")

try:
	minio_client = Minio(
		endpoint=MINIO_ENDPOINT,
		access_key=MINIO_ACCESS_KEY,
		secret_key=MINIO_SECRET_KEY,
		secure=False
	)
	
	if not minio_client.bucket_exists(MINIO_BUCKET_NAME):
		minio_client.make_bucket(MINIO_BUCKET_NAME)
		logging.info(f"创建桶Bucket成功: {MINIO_BUCKET_NAME}")
except Exception as e:
	logging.error(f"创建Minio客户端失败:{e}")
	minio_client = None
	
def get_minio_client()-> Minio:
	return minio_client