

# we need to install both plain python deps and torch...

# e.g. "cu128", "cpu" 
TORCH_WHEEL="cpu" 

# remove ald torch 
uv pip uninstall torch torchvision
# get plain dependencies
uv pip install -r pyproject.toml

uv pip install torch torchvision --index-url "https://download.pytorch.org/whl/$TORCH_WHEEL"

# things that depend on torch
uv pip install docling marker-pdf pymupdf4llm
