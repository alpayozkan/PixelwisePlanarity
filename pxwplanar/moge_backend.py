"""Import shim for the MoGe backend.

A repo checkout has the fork as the ``MoGe/`` submodule (imported as
``MoGe.moge``); a third-party install only packages ``pxwplanar*``, so the fork
comes in as the top-level ``moge`` distribution. Import the model symbols from
here and both work - the submodule wins when present.
"""

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent

if (_repo_root / "MoGe" / "moge").is_dir():
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    from MoGe.moge.model.v2 import MoGeModel, normalized_view_plane_uv
else:
    from moge.model.v2 import MoGeModel, normalized_view_plane_uv

__all__ = ["MoGeModel", "normalized_view_plane_uv"]
