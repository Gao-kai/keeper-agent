import os
from typing import Tuple
import logging
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

load_dotenv(override=True)

neo4j_driver = None

def get_neo4j_client(uri: str = "", auth: Tuple[str, str] = None, database: str = ""):
	global neo4j_driver
	if neo4j_driver is not None:
		return neo4j_driver
	
	try:
		neo4j_driver = GraphDatabase.driver(
			uri=uri or os.getenv("NEO4J_URI", ""),
			database=database or os.getenv("NEO4J_DATABASE", ""),
			auth=auth or (os.getenv("NEO4J_USERNAME", ""), os.getenv("NEO4J_PASSWORD", ""))
		)
		
		neo4j_driver.verify_connectivity()
		logging.info("连接NEO4J服务成功！")
		return neo4j_driver
	except Exception as e:
		logging.error(f"远程连接NEO4J服务失败{e}")
		return None
