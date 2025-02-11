import os
def find_project_root(target_project_name='CSIFormer'):
    """
    从 start_path 开始向上查找，直到找到文件夹名称与 target_project_name 相同的目录作为项目根目录。
    
    :param start_path: 查找起始路径
    :param target_project_name: 目标项目名称
    :return: 项目根目录的绝对路径
    :raises Exception: 如果遍历到文件系统根目录仍未找到，则抛出异常
    """
    current_path = os.path.dirname(os.path.abspath(__file__))
    while True:
        # 判断当前目录名称是否匹配目标项目名称
        if os.path.basename(current_path) == target_project_name:
            return current_path
        # 获取上级目录
        parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
        # 如果当前目录已经是文件系统的根目录，则停止查找
        if current_path == parent_path:
            raise Exception(f"未找到项目根目录：{target_project_name}")
        current_path = parent_path

# 从当前脚本所在的目录开始查找

project_root = find_project_root("CSIFormer")
print("项目根目录为:", project_root)