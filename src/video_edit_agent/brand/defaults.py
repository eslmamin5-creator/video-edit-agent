"""Default brand used when the user hasn't created one (spec section 23)."""
from __future__ import annotations

from video_edit_agent.brand.schema import Brand

DEFAULT_BRAND = Brand(name="default")


def default_brand_yaml() -> str:
    import yaml

    return yaml.safe_dump(DEFAULT_BRAND.model_dump(mode="json"), allow_unicode=True, sort_keys=False)
