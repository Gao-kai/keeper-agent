"""
创建Minio客户端
"""
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
		endpoint=MINIO_ACCESS_KEY,
		access_key="Q3AM3UQ867SPQQA43P2F",
		secret_key="zuf+tfteSlswRu7BJ86wekitnifILbZam1KYY3TG",
		secure=False
	)
	
	if not minio_client.bucket_exists(MINIO_BUCKET_NAME):
		minio_client.make_bucket(MINIO_BUCKET_NAME)
except Exception as e:
	print(f"创建Minio客户端失败:{e}")
	minio_client = None
	
def get_minio_client():
	return minio_client