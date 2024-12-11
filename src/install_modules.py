import subprocess
import sys
import os


def install_packages():
    try:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Define potential paths for the requirements.txt file
        requirements_paths = [
            os.path.join(script_dir, 'requirements.txt'),
            os.path.join(script_dir, '..', 'requirements.txt')
        ]

        # Find the correct path
        requirements_path = None
        for path in requirements_paths:
            if os.path.exists(path):
                requirements_path = path
                break

        if not requirements_path:
            raise FileNotFoundError("requirements.txt file not found")

        # Install packages
        with open(requirements_path, 'r') as file:
            packages = file.readlines()
            for package in packages:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package.strip()])

    except FileNotFoundError:
        print("requirements.txt file not found")
    except Exception as e:
        print(f"An error occurred: {e}")


def upgrade_pip():
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
        print()
    except Exception as e:
        print(f"An error occurred while upgrading pip: {e}")


if __name__ == '__main__':
    install_packages()
    upgrade_pip()
