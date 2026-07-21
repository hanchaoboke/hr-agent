# 迁移改造说明：Docker Compose + PostgreSQL + Redis

> 日期：2026-07-20
> 改造目标：
> 1. 引入 `docker-compose` 一键拉起基础设施，方便部署；
> 2. 员工数据库从本地 SQLite 迁移到 Docker 中的 PostgreSQL；
> 3. LangGraph 的会话检查点（checkpointer）从 `InMemorySaver` 换成 Redis，进程重启不再丢失对话记忆和挂起中的审批状态。

## 架构变化总览

```
改造前                                改造后
─────────────────────────            ─────────────────────────────────────
api.py / app.py                      api.py / app.py          （本机运行）
   │                                    │
   ├─ InMemorySaver（进程内存，          ├─ RedisSaver ──→ Redis Stack（Docker 容器）
   │   重启即丢）                        │                  · 对话多轮记忆
   │                                    │                  · 挂起的人工审批状态
   ├─ SQLite db/employee.db             ├─ psycopg2 ──→ PostgreSQL 16（Docker 容器）
   │  （本地文件）                        │                · employees / leave_balances
   │                                    │
   └─ Chroma db/chroma_db               └─ Chroma db/chroma_db（保持本地，未改动）
```

**范围说明**（有意为之的设计决策）：
- Docker 只容器化 **基础设施**（Postgres + Redis）。`api.py`、`app.py` 和 RAG 流水线仍在本机（Mac）运行——因为 RAG 使用了本地 BGE 嵌入/重排模型且指定 `device='mps'`（Apple 芯片 GPU），容器内既没有这些模型文件也没有 mps，强行容器化会导致 RAG 跑不起来或严重变慢。
- Chroma 向量库继续使用本地持久化目录 `db/chroma_db`，不迁入 Postgres。

---

## 逐文件改动明细（从底层到上层）

### 第 1 层 · 数据库层：`database/mock_db.py`（重写）

SQLite → PostgreSQL（psycopg2）。对外函数签名（`get_connection` / `init_db` / `query_db` / `close_db`）保持不变，上层代码几乎无感。

| 位置 | 改造前 | 改造后 | 原因 |
|------|--------|--------|------|
| 连接方式 | `sqlite3.connect(db_file, check_same_thread=False)`，本地文件 | `psycopg2.connect(**PG_CONFIG)`，配置全部来自环境变量 `POSTGRES_HOST/PORT/DB/USER/PASSWORD` | 连远程/容器数据库，配置外置方便部署 |
| 找不到库时 | 抛 `FileNotFoundError`，提示跑初始化脚本 | 抛 `ConnectionError`，提示先 `docker compose up -d` | 错误语义变了：不是文件丢失而是服务未启动 |
| 事务 | SQLite 默认 | `conn.autocommit = True` | 本项目查询为主，避免 psycopg2 默认开启事务导致长事务挂着 |
| 外键 | `PRAGMA foreign_keys=ON`（SQLite 专属） | 删除 | Postgres 外键默认强制，无需开关 |
| 线程安全 | `check_same_thread=False` 放开限制 | 新增模块级 `threading.Lock`，`query_db()` 内加锁 | `tools/hr_tools.py` 持有**单个模块级连接**，而 `api.py` 的后台超时线程会和请求线程并发用同一连接；psycopg2 单连接跨线程并发不安全，加锁串行化是最小且正确的改法 |
| SQL 占位符 | `?` | `%s` | psycopg2 的参数风格 |
| `init_db()` | 建表 + 灌种子（SQLite） | 逻辑不变：建表 + `DELETE` 清空 + 灌 4 名员工种子数据，保持幂等；只改了占位符和连接来源 | `python database/mock_db.py` 仍是初始化入口 |
| `query_db()` | 用 `cursor.description` 组装 dict 行 | 逻辑相同（psycopg2 同样支持），外加锁 | — |

### 第 2 层 · 工具层：`tools/hr_tools.py`（3 处小改）

只把 3 条 SQL 里的占位符 `?` 改成 `%s`：
- `get_employee_profile`：`where uid = %s`
- `check_leave_balance`：`where e.uid = %s`
- `generate_employment_certificate`：`where uid = %s`

其余（模块级 `db_conn = get_connection()`、`atexit` 清理钩子、返回文案）全部不动。

> ⚠️ 注意：`db_conn = get_connection()` 在 **import 时**执行，所以启动 api/app 前 Postgres 必须已就绪，否则 import 直接报 `ConnectionError`。

### 第 3 层 · Agent 层：`agent/graph_builder.py`（checkpointer 替换）

```python
# 改造前
from langgraph.checkpoint.memory import InMemorySaver
memory = InMemorySaver()
hr_agent_app = workflow.compile(checkpointer=memory)

# 改造后
import redis
from langgraph.checkpoint.redis import RedisSaver
_redis_client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
checkpointer = RedisSaver(redis_client=_redis_client)
checkpointer.setup()  # 首次运行创建 RediSearch 索引，幂等，必须调用一次
hr_agent_app = workflow.compile(checkpointer=checkpointer)
```

- 收益：**多轮对话记忆、挂起中的 human-in-the-loop 审批状态全部持久化**。uvicorn 重启后用同一 `thread_id` 继续对话，Agent 仍记得上文（已实测验证，见 `api_testing.md` 用例 8）。
- `RedisSaver` 依赖 **RediSearch + RedisJSON 模块**，所以 Redis 必须用 `redis-stack` 系列镜像（或 Redis 8.0+），普通 `redis` 镜像会在 `setup()` 建索引时报错。
- `api.py` / `app.py` 对 `hr_agent_app` 的调用方式（`.stream` / `.get_state` / `Command(resume=...)`）不需要因为换 checkpointer 而改——它是被 `compile()` 封装的实现细节。

### 第 3.5 层 · 顺带修复的 BUG：`api.py` / `app.py` 读取 interrupt 的方式

测试中发现一个**与本次迁移无关的既有 BUG**（迁移前的代码在当前 langgraph 1.2.4 + RedisSaver 组合下必现/高概率触发）：

- 原代码：流结束后用 `hr_agent_app.get_state(config)` 拿快照，再读 `state_snapshot.tasks[0].interrupts[0].value`。
- 问题：图挂起后**立刻** `get_state()`，`tasks[0].interrupts` 经常是空元组 `()`（checkpointer 的 pending-write 有读滞后），于是 `interrupts[0]` 抛 `IndexError: tuple index out of range`，接口返回 500。
- 修复（`api.py`）：改为在 stream 过程中直接捕获 interrupt——`stream_mode=["values", "updates"]` 双订阅，`updates` 流里出现 `__interrupt__` 键时取 `chunk["__interrupt__"][0].value`。这是官方推荐方式，不依赖落库时序，实测连续多轮 100% 稳定。
- 修复（`app.py`）：Streamlit 的页面刷新模式必须依赖 `get_state()`（跨 rerun 无法保留 stream 现场），所以改为优先读顶层 `state_snapshot.interrupts[0].value`（比 `tasks[0].interrupts` 可靠），为空时给出通用授权提示文案兜底。

### 第 4 层 · 配置：`.env` / `.env.sample`（追加）

```ini
# PostgreSQL（员工数据库，由 docker-compose 提供）
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=hr_agent
POSTGRES_USER=hr
POSTGRES_PASSWORD=hr_password

# Redis（LangGraph checkpointer，需 redis-stack，由 docker-compose 提供）
REDIS_URL=redis://localhost:6379
```

### 第 5 层 · 依赖：`requirements.txt`（追加）

```
psycopg2-binary            # Postgres 驱动（实测装的 2.9.12）
redis                      # Redis 客户端（7.4.1，注意会替换掉原来的 redis 8.0.0）
langgraph-checkpoint-redis # LangGraph Redis checkpointer（0.5.1）
```

### 第 6 层 · 部署：`docker-compose.yml`（新建）

两个服务，均带命名数据卷和健康检查：

| 服务 | 镜像 | 端口 | 数据卷 | 健康检查 |
|------|------|------|--------|----------|
| `postgres` | `postgres:16` | 5432 | `pgdata:/var/lib/postgresql/data` | `pg_isready` |
| `redis` | `redis/redis-stack:latest` | 6379 | `redisdata:/data` | `redis-cli ping` |

- Postgres 的库名/用户/密码从 `.env` 注入（compose 的 `${VAR:-default}` 语法，与应用共用同一份 `.env`）。
- Redis 用 `redis/redis-stack`（含 RediSearch/JSON）。如果你的环境网络通畅，也可以换成更精简的 `redis/redis-stack-server`——本机因 Docker Hub 拉取超时，改用了本地已有的 `redis-stack` 镜像，二者对本项目功能等价。

---

## 部署与启动步骤

```bash
# 1. 启动基础设施（首次会创建数据卷）
docker compose up -d
docker compose ps          # 等两个服务都显示 (healthy)

# 2. 首次部署：建表 + 灌种子数据（幂等，可重复执行）
python database/mock_db.py

# 3. 启动 API 服务（或 streamlit run app.py）
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

停止：`docker compose down`（保留数据）；`docker compose down -v`（连数据一起清掉）。

> 启动顺序很重要：**必须先等 Postgres healthy 再起 api/app**，因为 `tools/hr_tools.py` 在 import 时就建立数据库连接。

## 验证结果（实测）

1. ✅ `docker compose ps`：`hr_postgres`、`hr_redis` 均 `Up (healthy)`。
2. ✅ `python database/mock_db.py` 初始化成功；`psql` 查询 `employees` 表有 1001~1004 共 4 条种子数据。
3. ✅ `import agent.graph_builder` 成功：RedisSaver 索引创建、BGE 模型加载、混合索引构建全部正常。
4. ✅ API 全部接口用例通过（详见 `docs/api_testing.md`）。
5. ✅ **重启持久化**：kill 掉 uvicorn 再重启，用同一 `thread_id` 提问"你还记得我的职级吗"，Agent 直接答出 P4，无需重查数据库——证明状态在 Redis 而非进程内存。
