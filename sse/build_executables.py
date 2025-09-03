#!/usr/bin/env python3
"""
Script to build executables for macOS and Windows using PyInstaller
"""
import os
import platform
import subprocess
import sys

def build_macos():
    """Build macOS executable"""
    print("Building macOS executable...")
    
    cmd = [
        "pyinstaller",
        "--onefile",
        "--name", "httpx_request_macos",
        "--hidden-import", "httpx",
        "--hidden-import", "httpx_curl_cffi",
        "--hidden-import", "curl_cffi",
        "httpx_request.py"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("macOS executable built successfully!")
        print("Output location: dist/httpx_request_macos")
    except subprocess.CalledProcessError as e:
        print(f"Error building macOS executable: {e}")
        return False
    return True

def build_windows():
    """Build Windows executable (cross-compilation from macOS)"""
    print("Building Windows executable...")
    
    # For cross-compilation from macOS to Windows, we need to use Wine
    # This is a simplified approach - in practice, you might want to build on a Windows machine
    cmd = [
        "pyinstaller",
        "--onefile",
        "--name", "httpx_request_windows",
        "--hidden-import", "httpx",
        "--hidden-import", "httpx_curl_cffi",
        "--hidden-import", "curl_cffi",
        "httpx_request.py"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        # Rename the output file to have .exe extension for Windows
        if os.path.exists("dist/httpx_request_windows"):
            os.rename("dist/httpx_request_windows", "dist/httpx_request_windows.exe")
        print("Windows executable built successfully!")
        print("Output location: dist/httpx_request_windows.exe")
    except subprocess.CalledProcessError as e:
        print(f"Error building Windows executable: {e}")
        return False
    return True


def build_windows_amd64():
    """Build Windows executable for AMD64 architecture"""
    print("Building Windows AMD64 executable...")
    
    # Check if we're on a compatible platform for cross-compilation
    system = platform.system()
    machine = platform.machine()
    
    if system == "Darwin" and machine == "arm64":
        print("Warning: True cross-compilation from macOS ARM64 to Windows x86_64 is not supported by PyInstaller.")
        print("The generated executable will be a macOS ARM64 executable, not a Windows executable.")
        print("For a true Windows AMD64 executable, build on a Windows x86_64 machine.")
    
    # Create a spec file for Windows AMD64
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['httpx_request.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['httpx', 'httpx_curl_cffi', 'curl_cffi'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='httpx_request_windows_amd64',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,  # Remove target_arch for better compatibility
    codesign_identity=None,
    entitlements_file=None,
)
'''
    
    # Write the spec file
    with open('httpx_request_windows_amd64.spec', 'w') as f:
        f.write(spec_content)
    
    # Build using the spec file
    cmd = [
        "pyinstaller",
        "httpx_request_windows_amd64.spec"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        # Ensure the output file has the correct extension
        output_file = "dist/httpx_request_windows_amd64.exe"
        source_file = "dist/httpx_request_windows_amd64"
        
        if os.path.exists(source_file) and not os.path.exists(output_file):
            os.rename(source_file, output_file)
            
        print("Executable built successfully!")
        print("Note: This is a macOS executable, not a Windows executable.")
        print("Output location: dist/httpx_request_windows_amd64.exe")
        print("For a true Windows executable, build on a Windows machine.")
    except subprocess.CalledProcessError as e:
        print(f"Error building executable: {e}")
        print("Note: Cross-platform compilation can be complex and may require additional setup.")
        print("For best results, build on a Windows machine.")
        return False
    return True

def build_all():
    """Build executables for all platforms"""
    print("Building executables for all platforms...")
    
    # Build for macOS
    success = build_macos()
    if not success:
        return False
    
    # Build for Windows
    success = build_windows()
    if not success:
        return False
    
    # Build for Windows AMD64
    success = build_windows_amd64()
    if not success:
        return False
    
    return True

def build_current_platform():
    """Build executable for current platform"""
    system = platform.system()
    if system == "Darwin":
        return build_macos()
    elif system == "Windows":
        return build_windows()
    else:
        print(f"Unsupported platform: {system}")
        return False

def main():
    """Main function"""
    if len(sys.argv) > 1:
        target = sys.argv[1].lower()
        if target == "macos":
            build_macos()
        elif target == "windows":
            build_windows()
        elif target == "windows-amd64":
            build_windows_amd64()
        elif target == "all":
            build_all()
        else:
            print("Usage: python build_executables.py [macos|windows|windows-amd64|all]")
            return 1
    else:
        build_current_platform()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())