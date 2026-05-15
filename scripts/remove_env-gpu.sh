set -e

if [ ! -f "env-gpu.yaml" ]; then
    echo "❌ Error: env-gpu.yaml not found in the current directory."
    exit 1
fi

ENV_NAME=$(grep -E '^name:' env-gpu.yaml | awk '{print $2}')

if [ -z "$ENV_NAME" ]; then
    echo "❌ Error: Could not find a name in env-gpu.yaml."
    exit 1
fi

OS_TYPE="unknown"
case "$(uname -s)" in
    Linux*|Darwin*)
        OS_TYPE="unix"
        ;;
    CYGWIN*|MINGW*|MSYS*)
        OS_TYPE="windows"
        ;;
esac

echo "☢️  Starting total destruction of environment: $ENV_NAME ..."

ENV_PATH=$(conda info --envs | grep -w "$ENV_NAME" | awk '{print $NF}')

conda deactivate 2>/dev/null || true

echo "🔫 Killing any zombie Python or Ruff processes..."
if [ "$OS_TYPE" = "windows" ]; then
    taskkill //F //IM python.exe //T 2>/dev/null || true
    taskkill //F //IM ruff.exe //T 2>/dev/null || true
else
    pkill -9 python 2>/dev/null || true
    pkill -9 ruff 2>/dev/null || true
fi

echo "📦 Running conda remove..."
conda remove -n "$ENV_NAME" --all -y --force || true

if [ -n "$ENV_PATH" ] && [ -d "$ENV_PATH" ]; then
    echo "📂 Forcefully removing directory: $ENV_PATH"
    rm -rf "$ENV_PATH"

    if [ -d "$ENV_PATH" ]; then
        echo "❌ Folder still exists! Attempting native fallback delete..."

        if [ "$OS_TYPE" = "windows" ]; then
            powershell.exe -Command "Remove-Item -Recurse -Force -ErrorAction SilentlyContinue '$ENV_PATH'" 2>/dev/null || true
        else
            echo "⚠️ Could not delete $ENV_PATH. You may need manual intervention (e.g., 'sudo rm -rf $ENV_PATH')."
        fi
    fi
else
    echo "ℹ️  Environment folder already gone or not found."
fi

echo "🧹 Cleaning global index caches..."
conda clean -i -y

echo "✨ Environment '$ENV_NAME' has been scrubbed."
