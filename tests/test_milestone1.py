import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# 从tools包中导入函数
from tools.hr_tools import get_employee_profile, check_leave_balance, generate_employment_certificate

if __name__ == "__main__":
    print('--------- 测试1:查看 张三的档案---------')
    print(get_employee_profile.invoke({'uid':'1001'}))

    print('--------- 测试2:查看 李四的假期余额---------')
    print(check_leave_balance.invoke({'uid':'1002'}))

    print('--------- 测试3:查看 张三（P5）收入证明（预期成功）---------')
    print(generate_employment_certificate.invoke({'uid':'1001', 'cert_type':'income'}))

    print('--------- 测试4:查看 李四（P4）收入证明（预期：被拦截）---------')
    print(generate_employment_certificate.invoke({'uid': '1002', 'cert_type': 'income'}))
