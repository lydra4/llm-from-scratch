set -e

eval "$(mamba shell hook --shell bash)"

echo "🚀 Creating environment from env-cpu.yaml ..."
mamba env create -f env-cpu.yaml -y
echo "✅ Environment created."

ENV_NAME=$(grep -E '^name:' env-cpu.yaml | awk '{print $2}')

echo "🔧 Activating environment: $ENV_NAME ..."
mamba activate "$ENV_NAME"

echo "🛠️ Setting up pre-commit..."
pre-commit install
pre-commit run --all-files || true
echo "✅ Pre-commit setup completed."

echo "🧹 Cleaning conda & pip caches..."
mamba clean -a -y
conda clean --index-cache -y || true
pip cache purge
pre-commit clean
echo "✨ Cleanup done."
