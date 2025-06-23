# -*- coding: utf-8 -*-
"""
这个脚本会递归地扫描一个项目目录，查找一个预定义的关键字列表。
它会在文件名、目录名以及文件内容中进行不区分大小写的搜索，
并报告所有找到的匹配项。

它会自动忽略 .git, venv, 和 __pycache__ 目录。
"""
import sys
import re
from pathlib import Path

# --- 配置 ---

# 要查找的关键字列表（将进行不区分大小写的搜索）
# 包含空格的关键字也是支持的
KEYWORDS_TO_SEARCH = [
    'ztao8',
    'Music Being',
]

# 要在扫描中完全忽略的目录名称
EXCLUDED_DIRS = ['.git', 'venv', '__pycache__']

def search_project(root_dir: Path, pattern: re.Pattern):
    """
    在指定目录中查找关键字。

    返回:
        (list, list): 一个元组，包含两个列表：
                      1. 在名称中找到匹配项的列表。
                      2. 在内容中找到匹配项的列表。
    """
    found_in_names = []
    found_in_content = []

    print(f"[*] 正在扫描目录: {root_dir}")
    # 使用 rglob 递归查找所有文件和目录
    all_paths = list(root_dir.rglob('*'))

    for path in all_paths:
        # 检查路径中是否包含任何需要排除的目录
        if any(excluded in path.parts for excluded in EXCLUDED_DIRS):
            continue

        # 1. 检查文件名或目录名
        if pattern.search(path.name):
            found_in_names.append(path)

        # 2. 如果是文件，检查文件内容
        if path.is_file():
            try:
                # 尝试用 UTF-8 读取文件
                content = path.read_text(encoding='utf-8')
                # 逐行查找
                for i, line in enumerate(content.splitlines()):
                    # 使用 finditer 找到一行中的所有匹配
                    matches = list(pattern.finditer(line))
                    if matches:
                        # 为这一个文件只报告一次，但可以指出首次出现的行
                        found_in_content.append((path, i + 1, line.strip()))
                        # break可以确保每个文件只被报告一次，即使有多行匹配
                        break 
            except (UnicodeDecodeError, IOError):
                # 如果文件不是 UTF-8 编码或无法读取，则跳过内容搜索
                continue
    
    return found_in_names, found_in_content

def print_report(root_dir: Path, names: list, contents: list):
    """打印最终的查找报告。"""
    print("\n" + "="*20 + " 查找报告 " + "="*20)

    if not names and not contents:
        print("\n[✔] 未在任何文件名、目录名或文件内容中找到指定的关键字。")
        return

    if names:
        print("\n--- 在以下文件名或目录名中找到关键字 ---")
        for path in sorted(list(set(names))): # 去重并排序
            print(f"- {path.relative_to(root_dir)}")
    
    if contents:
        print("\n--- 在以下文件内容中找到关键字 ---")
        unique_content_paths = sorted(list(set(path for path, _, _ in contents))) # 只显示不重复的文件路径
        for path in unique_content_paths:
             # 找到这个路径的首次匹配信息
            _, line_num, line_text = next(item for item in contents if item[0] == path)
            print(f"- 文件: {path.relative_to(root_dir)}")
            print(f"  首次出现于第 {line_num} 行: \"{line_text}\"")

    print("\n" + "="*52)


def main():
    """主函数，用于运行脚本。"""
    if len(sys.argv) != 2:
        print("用法: python find_keywords.py <要扫描的目录>")
        print("示例: python find_keywords.py .")
        sys.exit(1)
        
    target_dir = Path(sys.argv[1]).resolve()
    
    if not target_dir.is_dir():
        print(f"错误: 指定的路径不是一个有效的目录: '{target_dir}'")
        sys.exit(1)

    # 创建一个不区分大小写的正则表达式
    # 例如: (ztao8|Music Being)
    regex_pattern = re.compile('|'.join(KEYWORDS_TO_SEARCH), re.IGNORECASE)

    found_names, found_contents = search_project(target_dir, regex_pattern)
    print_report(target_dir, found_names, found_contents)

if __name__ == "__main__":
    main()
