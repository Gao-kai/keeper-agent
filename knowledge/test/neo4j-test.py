"""
Neo4J 图数据库
1. 理解图数据库和向量数据库的区别以及在RAG中扮演的用途
2. 了解Neo4j中的节点、标签、关系以及属性的概念
3. 学会使用Docker-compose启动Neo4j本地服务
4. 增删改查操作
5. 完成电商图谱的案例
6. 事务保证要么全部成功要么全部失败

"""
import os
from typing import Optional, Tuple
from knowledge.processor.import_process.config import get_import_config
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
from dotenv import load_dotenv

load_dotenv(override=True)

neo4j_driver = None


def create_neo4j_client(uri: str, auth: Tuple[str, str], database: str):
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
		print("连接成功！")
		return neo4j_driver
	except Exception as e:
		raise Neo4jError(f"远程连接NEO4J服务失败{e}")
	
# 3. 创建分类节点
def create_category_node(neo4j_driver):
	
	categories = [
		"电子产品",
		"食品饮料",
		"服装鞋帽"
	]
	
	try:
		for category in categories:
			neo4j_driver.execute_query(
				query_="""
					CREATE (c:Category {name: $name})
					""",
				parameters_={"name": category},
				database_=database
			)
		
		print("✅分类节点创建完成")
	except Exception as e:
		raise Neo4jError(f"分类节点创建失败：{e}")

# 4. 创建商品节点并关联每一个商品的分类节点
def create_product_node_and_relate(neo4j_driver):
	products = [
		{"name": "华为手机", "price": 5999, "cat": "电子产品"},
		{"name": "进口咖啡", "price": 89, "cat": "食品饮料"},
		{"name": "运动跑鞋", "price": 599, "cat": "服装鞋帽"},
		{"name": "有机牛奶", "price": 168, "cat": "食品饮料"},
	]
	
	for product in products:
		neo4j_driver.execute_query(
			query_="""
			CREATE (p:Product {name:$name, price:$price, cat:$cat})
			WITH p
			MATCH (c:Category {name:$cat})
			CREATE (p)-[r:BELONGS_TO]->(c)
			""",
			parameters_={
				"name": product.get("name"),
				"price": product.get("price"),
				"cat": product.get("cat")
			},
			database_=database
		)
	
	print("✅成功创建商品节点并关联每一个商品的分类节点")

# 5. 创建客户节点
def create_custom_nodes(neo4j_driver):
	customers = [
		{"name": "张三", "age": 28, "vip": True},
		{"name": "李四", "age": 35, "vip": True},
		{"name": "王五", "age": 22, "vip": False},
		{"name": "小红", "age": 26, "vip": False},
	]
	
	for custom in customers:
		neo4j_driver.execute_query(
			query_="""
			CREATE (c:Custom {name:$name, age:$age, vip:$vip})
			""",
			parameters_={
				"name": custom.get("name"),
				"age": custom.get("age"),
				"vip": custom.get("vip")
			}
		)
	print("✅客户节点创建成功")

# 6. 建立客户和商品之间的购买关系以及购买价格
def create_custom_product_relation(neo4j_driver):
	relations = [
		("张三", "华为手机", 5999),
		("张三", "进口咖啡", 89),
		("李四", "华为手机", 5999),
		("李四", "有机牛奶", 168),
		("王五", "运动跑鞋", 599),
		("小红", "进口咖啡", 89),
		("小红", "运动跑鞋", 599),
	]
	for relation in relations:
		custom_name, product_name, price = relation
		neo4j_driver.execute_query(
			query_="""
			MATCH (c:Custom {name:$custom_name}),(p:Product {name:$product_name})
			WITH c,p
			CREATE (c)-[r:BUY {price: $price}]->(p)
			""",
			parameters_={
				"custom_name": custom_name,
				"product_name": product_name,
				"price": price
			}
		)
	print("✅ 成功建立客户和商品之间的购买关系以及购买价格")

# 7. 建立客户之间朋友关系
def create_friends_relations(neo4j_driver):
	friends = [
		("张三", "李四"),
		("张三", "王五"),
		("王五", "小红")
	]
	
	for friend in friends:
		a_name, b_name = friend
		neo4j_driver.execute_query(
			query_="""
			MATCH (c1:Custom {name: $a_name}),(c2:Custom {name: $b_name})
			WITH c1,c2
			CREATE (c1)-[r:FRIEND]->(c2)
			""",
			parameters_={
				"a_name": a_name,
				"b_name": b_name
			}
		)
	
	print("✅成功 建立客户之间朋友关系")


if __name__ == "__main__":
	config = get_import_config()
	# 1. 建立数据库Neo4j连接
	neo4j_driver = create_neo4j_client(
		uri=config.neo4j_uri,
		auth=(config.neo4j_username, config.neo4j_password),
		database=config.neo4j_database
	)
	database = config.neo4j_database
	
	# 2. 清空旧数据
	neo4j_driver.execute_query("""MATCH (n) DETACH DELETE (n)""")

	# 创建操作
	create_category_node(neo4j_driver)
	create_product_node_and_relate(neo4j_driver)
	create_custom_nodes(neo4j_driver)
	create_custom_product_relation(neo4j_driver)
	create_friends_relations(neo4j_driver)