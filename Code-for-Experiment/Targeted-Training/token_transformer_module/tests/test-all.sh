SCRIPT_DIRECTORY=$(dirname "$0")

python -m unittest discover -s $SCRIPT_DIRECTORY/scripts -p "test_*.py" "$@"