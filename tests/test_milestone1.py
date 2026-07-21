import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# 从tools包中导入函数
from tools.hr_tools import get_employee_profile, check_leave_balance, generate_employment_certificate


# =====================================================================
# pytest 用例：函数名必须以 test_ 开头，用 assert 断言判断对错
# 运行方式：pytest tests/test_milestone1.py -v
# =====================================================================
def test_get_employee_profile():
    """测试1：查询张三(1001)的档案，应包含姓名与职级"""
    result = get_employee_profile.invoke({'uid': '1001'})
    assert '张三' in result
    assert 'P5' in result


def test_check_leave_balance():
    """测试2：查询李四(1002)的假期余额，年假应为 7 天"""
    result = check_leave_balance.invoke({'uid': '1002'})
    assert '李四' in result
    assert '7' in result


def test_income_cert_p5_success():
    """测试3：张三(P5)开收入证明，预期成功"""
    result = generate_employment_certificate.invoke({'uid': '1001', 'cert_type': 'income'})
    assert '系统成功' in result
    assert '收入证明' in result


def test_income_cert_p4_blocked():
    """测试4：李四(P4)开收入证明，预期被职级权限拦截"""
    result = generate_employment_certificate.invoke({'uid': '1002', 'cert_type': 'income'})
    assert '无法' in result  # 返回文案里含“无法线上自助开具”


# =====================================================================
# 保留直接运行入口：python tests/test_milestone1.py 也能看到打印效果
# =====================================================================
if __name__ == "__main__":
    print('--------- 测试1:查看 张三的档案---------')
    print(get_employee_profile.invoke({'uid':'1001'}))

    print('--------- 测试2:查看 李四的假期余额---------')
    print(check_leave_balance.invoke({'uid':'1002'}))

    print('--------- 测试3:查看 张三（P5）收入证明（预期成功）---------')
    print(generate_employment_certificate.invoke({'uid':'1001', 'cert_type':'income'}))

    print('--------- 测试4:查看 李四（P4）收入证明（预期：被拦截）---------')
    print(generate_employment_certificate.invoke({'uid': '1002', 'cert_type': 'income'}))
