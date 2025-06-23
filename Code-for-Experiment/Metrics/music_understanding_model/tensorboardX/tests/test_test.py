def test_linting():
    import subprocess
    try:
        result = subprocess.run(['flake8', 'my_file.py'], capture_output=True, text=True, check=True)
        print("Linting successful!")
    except subprocess.CalledProcessError as e:
        print(f"Linting failed:\n{e.stderr}")