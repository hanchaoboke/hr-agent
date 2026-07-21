import atexit

from pathlib import Path
from langchain_core.tools import tool

from database.mock_db import get_connection, query_db, close_db

# 1. 获得数据库连接
db_conn = get_connection()
# 2. 注册进程退出时的清理钩子hook
atexit.register(close_db, db_conn)

@tool
def get_employee_profile(uid:str) -> str:
    """
    根据员工 UID 查询员工的完整人事档案，包括姓名、职级、城市、入职年限及基本薪资。
    当需要获得当前对话员工的背景属性时，必须首先调用此工具
    """
    sql = "select uid, name, rank, location, seniority, base_salary from employees where uid = %s"
    res = query_db(conn=db_conn, sql=sql, params=(uid,))
    if not res:
        return f'ERROR:未找到UID 为 {uid} 的员工信息。'

    emp = res[0]
    return (f'【档案查询结果】员工姓名：{emp['name']}，职级：{emp['rank']},'
            f'属地：{emp['location']},入职年限：{emp['seniority']}年,'
            f'基本薪资：{emp['base_salary']}元。')

@tool
def check_leave_balance(uid:str) -> str:
    """根据员工UID 查询其剩余假期余额（包括年假和病假）
    当员工明确询问‘我还有几天假’或‘我的假期余额’时使用"""
    sql = """
        select e.uid, e.name, l.annual_leave_remaining, l.sick_leave_remaining
        from employees e
        join leave_balances l on e.uid = l.uid
        where e.uid = %s
    """
    res = query_db(conn=db_conn, sql=sql, params=(uid,))
    if not res:
        return f'ERROR：无法获取uid为 {uid} 的假期数据'
    data = res[0]
    return (f"【假期系统】员工 {data['name']} （UID: {data['uid']}）当前剩余法定/福利年假：{data['annual_leave_remaining']} 天，"
            f"剩余带薪病假：{data['sick_leave_remaining']} 天。")

@tool
def generate_employment_certificate(uid:str, cert_type:str) -> str:
    """
    为指定员工自动生成带有电子签章的证明文件
    参数 cert_type 必须是以下两个值之一：
    - 'employment': 仅开具在职证明（全员可用）
    - 'income': 开具包含薪资的在职及收入证明（有职级权限限制，仅 P5 及以上可用）
    """
    emp_res = query_db(conn=db_conn, sql="select name, rank, base_salary from employees where uid = %s", params=(uid,))
    if not emp_res:
        return f'因无法核实员工身份（UID：{uid}），证明生成失效'
    emp = emp_res[0]
    if cert_type == 'income':
        try:
            rank_level = int(emp['rank'].replace('P',''))
        except ValueError:
            rank_level = 0

        if rank_level < 5:
            return (f"【系统提示】根据公司规章制度，P4及以下职级员工（当前员工职级为：{emp['rank']}）无法线上自助开具薪资收入证明。"
                    f"引导员工在线提交人工工单")

        content = (f"《薪资收入证明》\n兹证明我司员工 {emp['name']},职级为 {emp['rank']} 。该员工基本薪资为人民币 {emp['base_salary']}."
                   f"特此证明。 （盖章）")

        return f'【系统成功】以自动为您生成收入证明：\n------------\n{content}\n-------------------'

    elif cert_type == 'employment':
        content = (
            f"《在职证明》\n兹证明我司员工 {emp['name']},职级为 {emp['rank']} 。"
            f"特此证明。 （盖章）")

        return f'【系统成功】以自动为您生成在职证明：\n------------\n{content}\n-------------------'
    else:
        return f'ERROR:不支持的证明类型。可选为employment 或 income'