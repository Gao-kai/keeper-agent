
# 使用方式
# 进入项目目录后直接用 `make <命令>`：

## 命令速查表

#| 命令 | 功能 |
#|------|------|
#| `make up` | 启动所有服务（后台运行） |
#| `make down` | 停止并删除容器（保留数据） |
#| `make stop` | 仅停止服务 |
#| `make restart` | 重启所有服务 |
#| `make build` | 重新构建并启动 |
#| `make pull` | 拉取最新镜像 |
#| `make ps` | 查看容器状态 |
#| `make logs` | 查看所有服务日志 |
#| `make logs-mongo` | 查看 MongoDB 日志 |
#| `make logs-milvus` | 查看 Milvus 日志 |
#| `make mongo-shell` | 进入 MongoDB shell |
#| `make milvus-shell` | 进入 Milvus 容器 |
#| `make status` | 查看所有服务健康状态 |
#| `make clean` | 清理容器和数据卷（⚠️ 会删数据） |
#| `make help` | 显示帮助信息 |


# := 是 Makefile 的赋值运算符（立即赋值）
COMPOSE_FILE := milvus-standalone-docker-compose.yml
COMPOSE := docker compose -f $(COMPOSE_FILE)

.PHONY: up down stop restart ps logs logs-mongo logs-milvus \
        mongo-shell milvus-shell status build pull clean help

# 启动所有服务（后台运行）
up:
	$(COMPOSE) up -d

# 停止并删除容器（保留数据卷）
down:
	$(COMPOSE) down

# 仅停止服务（不删除容器）
stop:
	$(COMPOSE) stop

# 重启所有服务
restart:
	$(COMPOSE) restart

# 重新构建并启动服务
build:
	$(COMPOSE) up -d --build

# 拉取最新镜像
pull:
	$(COMPOSE) pull

# 查看容器运行状态
ps:
	$(COMPOSE) ps

# 查看所有服务日志（实时跟踪）
logs:
	$(COMPOSE) logs -f

# 查看 MongoDB 日志
logs-mongo:
	$(COMPOSE) logs -f mongodb

# 查看 Milvus 日志
logs-milvus:
	$(COMPOSE) logs -f standalone

# 进入 MongoDB shell
mongo-shell:
	docker exec -it mongo_db mongosh -u booker -p Manulife --authenticationDatabase admin

# 进入 Milvus 容器 shell
milvus-shell:
	docker exec -it shopkeeper-milvus-standalone bash

# 查看所有服务健康状态
status:
	@echo "========== 容器状态 =========="
	@$(COMPOSE) ps
	@echo ""
	@echo "========== MongoDB 健康 =========="
	@docker inspect --format='{{.State.Health.Status}}' mongo_db 2>/dev/null || echo "未运行"
	@echo ""
	@echo "========== Milvus 健康 =========="
	@docker inspect --format='{{.State.Health.Status}}' shopkeeper-milvus-standalone 2>/dev/null || echo "未运行"
	@echo ""
	@echo "========== Neo4j 健康 =========="
	@docker inspect --format='{{.State.Health.Status}}' neo4j 2>/dev/null || echo "未运行"

# 清理所有容器和数据卷（⚠️ 会删除所有数据！）
clean:
	$(COMPOSE) down -v

# 查看所有可用命令
help:
	@echo "可用命令:"
	@echo "  make up            - 启动所有服务（后台）"
	@echo "  make down          - 停止并删除容器（保留数据）"
	@echo "  make stop          - 仅停止服务"
	@echo "  make restart       - 重启所有服务"
	@echo "  make build         - 重新构建并启动"
	@echo "  make pull          - 拉取最新镜像"
	@echo "  make ps            - 查看容器状态"
	@echo "  make logs          - 查看所有服务日志"
	@echo "  make logs-mongo    - 查看 MongoDB 日志"
	@echo "  make logs-milvus   - 查看 Milvus 日志"
	@echo "  make mongo-shell   - 进入 MongoDB shell"
	@echo "  make milvus-shell  - 进入 Milvus 容器"
	@echo "  make status        - 查看所有服务健康状态"
	@echo "  make clean         - 清理容器和数据卷（⚠️ 危险）"
	@echo "  make help          - 显示此帮助信息"

