#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Cookie更新工具
帮助用户更新cookie.txt文件
"""

import os


def update_cookie():
    """交互式更新cookie"""
    print("=== Cookie更新工具 ===")
    print("\n获取Cookie的步骤：")
    print("1. 打开浏览器访问 https://lmarena.ai")
    print("2. 登录你的账户")
    print("3. 按F12打开开发者工具")
    print("4. 在Network标签页中刷新页面")
    print("5. 找到任意请求，右键选择'Copy' -> 'Copy request headers'")
    print("6. 从复制的内容中找到Cookie行，复制Cookie的值")
    print("\n注意：Cookie值通常很长，包含多个分号分隔的键值对")
    print("-" * 50)
    
    # 显示当前cookie状态
    if os.path.exists('cookie.txt'):
        try:
            with open('cookie.txt', 'r', encoding='utf-8') as f:
                current_cookie = f.read().strip()
                if current_cookie:
                    print(f"\n当前cookie.txt文件存在，长度: {len(current_cookie)} 字符")
                    print(f"前50个字符: {current_cookie[:50]}...")
                else:
                    print("\n当前cookie.txt文件为空")
        except Exception as e:
            print(f"\n读取当前cookie.txt时出错: {e}")
    else:
        print("\n当前不存在cookie.txt文件")
    
    print("\n选择更新方式:")
    print("1. 直接输入cookie (适合短cookie)")
    print("2. 从剪贴板读取cookie (推荐)")
    print("3. 退出")
    
    try:
        choice = input("请选择 (1/2/3): ").strip()
        
        if choice == '3':
            print("已取消更新")
            return
        elif choice == '2':
            # 尝试从剪贴板读取
            try:
                import subprocess
                # macOS剪贴板命令
                result = subprocess.run(['pbpaste'], capture_output=True, text=True)
                new_cookie = result.stdout.strip()
                
                if not new_cookie:
                    print("剪贴板为空，请先复制cookie到剪贴板")
                    return
                
                print(f"从剪贴板读取到cookie，长度: {len(new_cookie)} 字符")
                print(f"前50个字符: {new_cookie[:50]}...")
                
                confirm = input("确认使用此cookie? (y/N): ").strip().lower()
                if confirm != 'y':
                    print("已取消更新")
                    return
                    
            except Exception as e:
                print(f"从剪贴板读取失败: {e}")
                print("请选择方式1手动输入")
                return
        elif choice == '1':
            print("\n请输入cookie (输入'quit'退出):")
            new_cookie = input().strip()
        else:
            print("无效选择")
            return
        
        if new_cookie.lower() == 'quit':
            print("已取消更新")
            return
        
        if not new_cookie:
            print("错误: Cookie值不能为空")
            print("请确保已正确粘贴cookie并按回车")
            return
        
        # 基本验证
        if 'arena-auth-prod' not in new_cookie:
            print("警告: Cookie中似乎缺少arena-auth-prod认证信息")
            confirm = input("是否仍要保存? (y/N): ").strip().lower()
            if confirm != 'y':
                print("已取消更新")
                return
        
        # 保存cookie
        with open('cookie.txt', 'w', encoding='utf-8') as f:
            f.write(new_cookie)
        
        print(f"\n✅ Cookie已成功保存到cookie.txt")
        print(f"Cookie长度: {len(new_cookie)} 字符")
        print("\n现在可以重新运行httpx_request.py测试连接")
        
    except KeyboardInterrupt:
        print("\n\n已取消更新")
    except Exception as e:
        print(f"\n错误: 保存cookie时出错: {e}")

if __name__ == "__main__":
    update_cookie()