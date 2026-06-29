import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DB_FILE = PROJECT_ROOT / "db" / "employee.db"

def get_connection(db_file=DB_FILE) -> sqlite3.Connection:
    """业务运行时连接函数，仅建立连接并开启外键，不会写数据"""
    if not db_file.exists():
        raise FileNotFoundError(f"\n 错误，数据库文件未找到: {db_file}\n请先手动运行一下初始化脚本python database/mock_db.py")

    conn = sqlite3.connect(db_file)
    conn.execute('PRAGMA foreign_keys=ON')  # 启动 SQLite 外键功能
    return conn


def init_db(db_path:Path=DB_FILE)->sqlite3.Connection:
    """初始化数据库（仅供手动单次运行）"""
    # 1. 自动创建数据库db目录
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # 2. 连接数据库（若不存数据库文件自动创建）
    conn = sqlite3.connect(db_path, check_same_thread=False)    # check_same_thread 允许多线程使用
    conn.execute('PRAGMA foreign_keys=ON')
    cursor = conn.cursor()

    # 3. 创建员工表
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

    # 4. 创建假期表
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS leave_balances (
                uid TEXT PRIMARY KEY,                    -- 员工唯一标识（外键关联employees.uid）
                annual_leave_remaining INTEGER NOT NULL, -- 剩余年假天数
                sick_leave_remaining INTEGER NOT NULL,   -- 剩余病假天数
                FOREIGN KEY(uid) REFERENCES employees(uid)
            )
        """)

    # 5. 清空旧数据（确保幂等性）
    cursor.execute("delete from leave_balances")
    cursor.execute("delete from employees")

    # 6. 初始测试数据
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

    cursor.executemany("insert into employees values (?, ?, ?, ?, ?, ?)", test_employees)
    cursor.executemany("insert into leave_balances values (?, ?, ?)", test_balances)

    conn.commit()
    print('初始数据库以创建')
    print(f'数据库文件路径：{db_path}')
    return conn

def query_db(conn:sqlite3.Connection, sql:str, params:tuple=()):
    """通用数据查询"""
    cursor = conn.cursor()
    cursor.execute(sql, params)
    columns = [col[0] for col in cursor.description]
    return [dict(zip(columns, row))for row in cursor.fetchall()]

def close_db(conn:sqlite3.Connection):
    """安全关闭数据库"""
    if conn:
        conn.close()
        print('数据库连接以成功关闭')

if __name__ == '__main__':
    print('正在执行数据库手动初始化......')
    standalone = init_db()
    close_db(standalone)
