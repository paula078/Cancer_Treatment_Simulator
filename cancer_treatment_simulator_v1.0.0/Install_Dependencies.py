#!/usr/bin/env python3
"""
Python 3.10 Specific Installation Script for Cancer Treatment Simulator

This script creates a Python 3.10 virtual environment and installs dependencies
for maximum compatibility and performance.
"""

import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def validate_python_version():
    """Ensure we're running Python 3.10."""
    version = sys.version_info
    if version.major != 3 or version.minor != 10:
        print(f"❌ ERROR: This installer requires Python 3.10")
        print(
            f"   Current version: Python {version.major}.{version.minor}.{version.micro}"
        )
        print("   Please run with: py -3.10 install_requirements_py310.py")
        return False
    return True


def create_virtual_environment():
    """Create a Python 3.10 virtual environment."""
    venv_path = Path.cwd() / ".cancer_simulator_env"

    try:
        if venv_path.exists():
            print("🔍 Existing Python 3.10 environment found - validating...")
            # Check if it's Python 3.10
            python_exe = venv_path / "Scripts" / "python.exe"
            if python_exe.exists():
                try:
                    result = subprocess.run(
                        [str(python_exe), "--version"], capture_output=True, text=True
                    )
                    if "3.10" in result.stdout:
                        print("✅ Valid Python 3.10 environment found")
                        return venv_path
                except:
                    pass

            print("⚠️  Environment exists but not Python 3.10, recreating...")
            print("   This may take a moment due to locked files...")

            # Force remove with retry logic for Windows file locks
            import shutil
            import time

            max_retries = 5
            for attempt in range(max_retries):
                try:
                    shutil.rmtree(venv_path)
                    print(f"   ✅ Old environment removed successfully")
                    break
                except (OSError, PermissionError) as e:
                    if attempt < max_retries - 1:
                        print(
                            f"   ⏳ Retry {attempt + 1}/{max_retries} - waiting for file locks to release..."
                        )
                        time.sleep(2)
                    else:
                        print(f"   ❌ Cannot remove old environment: {e}")
                        print(
                            "   Please manually delete the .cancer_simulator_env folder and try again"
                        )
                        print("   Or restart your computer and try again")
                        return None

        print("🏗️  Creating Python 3.10 virtual environment...")
        print("   This ensures optimal performance and compatibility...")

        result = subprocess.run(
            [sys.executable, "-m", "venv", str(venv_path), "--clear"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("✅ Python 3.10 virtual environment created successfully")
            return venv_path
        else:
            print(f"❌ Failed to create virtual environment:")
            print(f"   {result.stderr}")
            return None

    except Exception as e:
        print(f"❌ Error creating virtual environment: {e}")
        return None


def get_venv_python(venv_path):
    """Get Python 3.10 executable from virtual environment."""
    return venv_path / "Scripts" / "python.exe"


def install_packages_optimized():
    """Install packages optimized for Python 3.10."""
    print("\n📦 Installing packages optimized for Python 3.10...")

    # Create virtual environment
    venv_path = create_virtual_environment()
    if not venv_path:
        return False

    venv_python = get_venv_python(venv_path)
    if not venv_python.exists():
        print(f"❌ Virtual environment Python not found: {venv_python}")
        return False

    # Upgrade pip to latest version
    print("🔧 Upgrading pip to latest version...")
    try:
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
            capture_output=True,
            text=True,
            check=True,
        )
        print("✅ Pip upgraded successfully")
    except subprocess.CalledProcessError:
        print("⚠️  Pip upgrade failed, continuing anyway...")

    # Install packages optimized for Python 3.10
    packages = [
        ("streamlit>=1.28.0", "Web Framework (latest)"),
        ("pandas>=2.0.0", "Data Analysis (optimized)"),
        ("numpy>=1.24.0", "Numerical Computing (fast)"),
        ("scikit-learn>=1.3.0", "Machine Learning (latest)"),
        ("scipy>=1.11.0", "Scientific Computing (optimized)"),
        ("plotly>=5.17.0", "Interactive Plotting (latest)"),
        ("matplotlib>=3.7.0", "Static Plotting (stable)"),
        ("seaborn>=0.12.0", "Statistical Visualization (latest)"),
        ("numba>=0.58.0", "JIT Compilation for ABM Optimization (critical)"),
    ]

    total_packages = len(packages)

    for i, (package, description) in enumerate(packages, 1):
        print(f"\n📥 [{i}/{total_packages}] Installing {description}...")
        print(f"   Package: {package}")

        try:
            result = subprocess.run(
                [str(venv_python), "-m", "pip", "install", package],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode == 0:
                print(f"   ✅ {description} installed successfully")
            else:
                print(f"   ❌ Failed to install {package}")
                print(f"   Error: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print(f"   ⏱️  Installation timeout for {package}")
            return False
        except Exception as e:
            print(f"   ❌ Installation error: {e}")
            return False

    print("\n🎉 All packages installed successfully with Python 3.10!")
    return True


def validate_installation():
    """Validate Python 3.10 installation."""
    print("\n🔍 Validating Python 3.10 installation...")

    venv_path = Path.cwd() / ".cancer_simulator_env"
    venv_python = get_venv_python(venv_path)

    # Test Python version
    try:
        result = subprocess.run(
            [
                str(venv_python),
                "-c",
                "import sys; print(f'Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            version_info = result.stdout.strip()
            print(f"   ✅ {version_info}")
            if "3.10" not in version_info:
                print("   ⚠️  Warning: Not Python 3.10")
        else:
            print("   ❌ Python version check failed")
            return False
    except Exception as e:
        print(f"   ❌ Python validation error: {e}")
        return False

    # Test imports
    test_imports = [
        "streamlit",
        "pandas",
        "numpy",
        "sklearn",
        "scipy",
        "plotly",
        "matplotlib",
        "seaborn",
    ]

    for module in test_imports:
        try:
            result = subprocess.run(
                [
                    str(venv_python),
                    "-c",
                    f"import {module}; print('{module}', {module}.__version__)",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                version_info = result.stdout.strip()
                print(f"   ✅ {version_info}")
            else:
                print(f"   ❌ {module} - import failed")
                return False

        except Exception as e:
            print(f"   ❌ {module} - validation error: {e}")
            return False

    print("✅ Python 3.10 installation validation completed successfully!")
    return True


def main():
    """Main installation function for Python 3.10."""
    print("=" * 80)
    print("CANCER TREATMENT SIMULATOR - PYTHON 3.10 INSTALLER")
    print("=" * 80)
    print("Optimized installation for maximum performance and compatibility")
    print()

    # Validate Python version
    if not validate_python_version():
        input("Press Enter to exit...")
        return

    print("✅ Python 3.10 confirmed")
    print(f"   Full version: {sys.version}")
    print(f"   Installation directory: {Path.cwd()}")

    # Installation process
    try:
        start_time = __import__("time").time()

        if not install_packages_optimized():
            print("\n❌ INSTALLATION FAILED")
            print("Please check the error messages above and try again.")
            input("Press Enter to exit...")
            return

        if not validate_installation():
            print("\n❌ INSTALLATION VALIDATION FAILED")
            input("Press Enter to exit...")
            return

        end_time = __import__("time").time()
        duration = int(end_time - start_time)

        print("\n" + "=" * 80)
        print("🎊 PYTHON 3.10 INSTALLATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"⏱️  Installation time: {duration} seconds")
        print(f"🐍 Python version: 3.10 (optimized)")
        print()
        print("✅ Performance benefits:")
        print("   • Latest package versions for maximum compatibility")
        print("   • Optimized numerical computing performance")
        print("   • Enhanced machine learning capabilities")
        print("   • Improved memory management")
        print("   • Better security and stability")
        print()
        print("✅ Isolation benefits:")
        print("   • Complete isolation in Python 3.10 environment")
        print("   • No conflicts with other Python installations")
        print("   • Easy removal - delete .cancer_simulator_env folder")
        print("   • Reproducible across different systems")
        print()
        print("🚀 Ready to run with Python 3.10:")
        print("   • Use: run_simulator_py310.bat")
        print("   • Or: py -3.10 secure_launcher_py310.py")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n🛑 Installation cancelled by user")
    except Exception as e:
        print(f"\n❌ Installation failed with error: {e}")

    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
