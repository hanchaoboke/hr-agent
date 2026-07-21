import os
import threading

import psycopg2
from dotenv import load_dotenv

load_dotenv()

# =====================================================================
# PostgreSQL 连接配置（全部走环境变量，方便 docker-compose 部署）
# =====================================================================
PG_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "dbname": os.getenv("POSTGRES_DB", "hr_agent"),
    "user": os.getenv("POSTGRES_USER", "hr"),
    "password": os.getenv("POSTGRES_PASSWORD", "hr_password"),
}

# hr_tools.py 持有单个模块级连接，而 api.py 的后台超时线程会与请求线程并发使用它。
# psycopg2 的单连接跨线程并发不安全，用一把模块级锁把查询串行化（最小改动且正确）。
_query_lock = threading.Lock()


def get_connection(config: dict = None) -> psycopg2.extensions.connection:
    """业务运行时连接函数，仅建立连接，不会写数据。

    连接失败时抛出清晰的错误，提示先用 docker-compose 拉起 Postgres。
    """
    cfg = config or PG_CONFIG
    try:
        conn = psycopg2.connect(**cfg)
        conn.autocommit = True  # 查询为主，开启自动提交避免长事务
        return conn
    except psycopg2.OperationalError as e:
        raise ConnectionError(
            f"\n 错误，无法连接 PostgreSQL（{cfg['host']}:{cfg['port']}/{cfg['dbname']}）：{e}\n"
            f"请先执行 `docker compose up -d` 拉起数据库，并确认 .env 中的 POSTGRES_* 配置正确。"
        ) from e


def init_db(config: dict = None) -> psycopg2.extensions.connection:
    """初始化数据库（仅供手动单次运行）：建表 + 灌入种子数据。"""
    conn = get_connection(config)
    cursor = conn.cursor()

    # 1. 创建员工表
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS employees (
                uid TEXT PRIMARY KEY,                    -- 员工唯一标识
                name TEXT NOT NULL,                      -- 员工姓名
                rank TEXT NOT NULL,                      -- 职级（如P3、P4、P5等）
                location TEXT NOT NULL,                  -- 工作地点（城市名称）
                seniority INTEGER NOT NULL,              -- 入职年限（年）
                base_salary INTEGER NOT NULL             -- 基本工资（元）
            )
        """)

    # 2. 创建假期表
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS leave_balances (
                uid TEXT PRIMARY KEY,                    -- 员工唯一标识（外键关联employees.uid）
                annual_leave_remaining INTEGER NOT NULL, -- 剩余年假天数
                sick_leave_remaining INTEGER NOT NULL,   -- 剩余病假天数
                FOREIGN KEY(uid) REFERENCES employees(uid)
            )
        """)

    # 3. 清空旧数据（确保幂等性）；先删子表再删主表以满足外键约束
    cursor.execute("delete from leave_balances")
    cursor.execute("delete from employees")

    # 4. 初始测试数据
    test_employees = [
        ("1001", "张三", "P5", "北京", 2, 18000),  # P5, 一线城市, 入职2年
        ("1002", "李四", "P4", "成都", 4, 9000),  # P4, 二线城市, 入职4年
        ("1003", "王五", "P7", "上海", 5, 35000),  # P7, 一线城市, 入职5年
        ("1004", "赵六", "P3", "深圳", 0, 7500)  # P3, 新入职不满1年
    ]

    test_balances = [
        ("1001", 6, 10),  # 张三: 年假剩余6天, 病假剩余10天
        ("1002", 7, 12),  # 李四: 年假剩余7天, 病假剩余12天
        ("1003", 14, 15),  # 王五: 年假剩余14天, 病假剩余15天
        ("1004", 2, 5)  # 赵六: 年假剩余2天, 病假剩余5天
    ]

    cursor.executemany("insert into employees values (%s, %s, %s, %s, %s, %s)", test_employees)
    cursor.executemany("insert into leave_balances values (%s, %s, %s)", test_balances)

    cursor.close()
    print('初始数据库以创建')
    print(f'数据库地址：{PG_CONFIG["host"]}:{PG_CONFIG["port"]}/{PG_CONFIG["dbname"]}')
    return conn


def query_db(conn: psycopg2.extensions.connection, sql: str, params: tuple = ()):
    """通用数据查询（占位符用 %s）。加锁串行化以适配跨线程共享连接。"""
    with _query_lock:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        columns = [col[0] for col in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        cursor.close()
        return rows


def close_db(conn: psycopg2.extensions.connection):
    """安全关闭数据库"""
    if conn:
        conn.close()
        print('数据库连接以成功关闭')


if __name__ == '__main__':
    print('正在执行数据库手动初始化......')
    standalone = init_db()
    close_db(standalone)
