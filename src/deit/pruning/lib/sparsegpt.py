"""Load the pinned RIA dependency installed by third_party/deit/setup_ria.py."""
from ._external import load

load('sparsegpt.py', globals())
