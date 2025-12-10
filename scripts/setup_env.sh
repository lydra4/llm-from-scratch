set -e

eval "$(conda shell.bash hook)"

echo "🚀 Creating environment from env.yaml ..."
mamba env create -f env.yaml -y
echo "✅ Environment created."

# Extract env name from env.yaml
ENV_NAME=$(grep -E '^name:' env.yaml | awk '{print $2}')

echo "🔧 Activating environment: $ENV_NAME ..."
conda activate "$ENV_NAME"

echo "🛠️ Setting up pre-commit..."
pre-commit install
pre-commit run --all-files || true
echo "✅ Pre-commit setup completed."

echo "🧹 Cleaning conda & pip caches..."
mamba clean -a -y
pip cache purge
echo "✨ Cleanup done."
