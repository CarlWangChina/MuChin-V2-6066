SCRIPT_DIRECTORY=$(dirname "$0")

if [ -z "$1" ]; then
    echo "Usage: $0 <test_python_file> [unittest_args]"
    exit 1
fi

python -m unittest discover -s $SCRIPT_DIRECTORY/scripts -p "$1" "${@:2}"