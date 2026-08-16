import os.path


PROJECT_ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))

LOCAL_CACHE_DIR_NAME = os.path.join(PROJECT_ROOT_PATH,'cache')

FRONT_STATIC_RESOURCE_DIR_NAME = os.path.join(PROJECT_ROOT_PATH,'ui','dist')


def get_local_cache_dir_name():
	return LOCAL_CACHE_DIR_NAME

def get_static_dir_name():
	return FRONT_STATIC_RESOURCE_DIR_NAME

if __name__ == "__main__":
	print(PROJECT_ROOT_PATH)
	print(LOCAL_CACHE_DIR_NAME)
	print(FRONT_STATIC_RESOURCE_DIR_NAME)