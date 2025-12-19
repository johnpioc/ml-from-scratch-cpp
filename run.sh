set -e
BUILD_DIR="build"
CPP_EXECUTABLE_NAME="ml_from_scratch" 

if [ -z "$1" ]; then
    echo "Usage: ./run.sh [model]"
    echo ""
    echo "Please provide the model that you want to run: "
    echo "Linear Regression: -linReg"
    echo "Logistic Regression: -logReg"
    exit 1
fi

MODEL_TYPE=$1
PYTHON_SCRIPT_PATH=""

case $MODEL_TYPE in
  "-linReg")
    PYTHON_SCRIPT_PATH="src/benchmark/LinearRegression.py"
    ;;
  "-logReg")
    PYTHON_SCRIPT_PATH="src/benchmark/LogisticRegression.py"
    ;;
  *)
    echo "Error: Unknown model type '$MODEL_TYPE'"
    echo "Available options: linear, logistic"
    exit 1
    ;;
esac

if [ ! -d "$BUILD_DIR" ]; then
  mkdir $BUILD_DIR
fi 

VENV_DIR=".venv"
REQUIREMENTS_FILE="requirements.txt"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    
    if [ -f "$REQUIREMENTS_FILE" ]; then
        echo "📦 Installing dependencies from $REQUIREMENTS_FILE..."
        ./$VENV_DIR/bin/pip install -r "$REQUIREMENTS_FILE" > /dev/null
    else
        echo "⚠️  Warning: $REQUIREMENTS_FILE not found!"
    fi
fi

DATA_ZIP="data.zip"
DATA_DIR="data" 

if [ -f "$DATA_ZIP" ]; then
    if [ -d "$DATA_DIR" ]; then
        rm -rf "$DATA_DIR"
    fi
    
    unzip -q -o "$DATA_ZIP"
else
    echo "⚠️  Warning: $DATA_ZIP not found. Using existing data (if any)."
fi

cd $BUILD_DIR

cmake .. > /dev/null
make > /dev/null

echo "✅ Build Successful."
echo ""

./$CPP_EXECUTABLE_NAME $MODEL_TYPE
echo ""

cd ..
./$VENV_DIR/bin/python "$PYTHON_SCRIPT_PATH"
