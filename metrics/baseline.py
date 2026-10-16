from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from config.models_config import resolve_deltae_json
from metrics.kendall_tau_ranking import (
    augment_meta,
    load_deltae_override,
    load_metadata_index,
)


def prepare_meta(
    meta_path: Path,
    *,
    baseline: str = "gfn2-xtb",
    deltae_json: Path | None = None,
) -> dict[str, dict[str, Any]]:
    resolved = resolve_deltae_json(baseline, deltae_json)
    override = load_deltae_override(resolved)
    return augment_meta(
        load_metadata_index(meta_path),
        override,
        strict=override is not None,
    )
