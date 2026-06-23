"""
Bhoonidhi STAC search scaffold (NRSC/ISRO).

API spec: https://bhoonidhi.nrsc.gov.in/bhoonidhi-api/
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests

BHOO_STAC_ROOT = os.environ.get(
    "BHOO_STAC_ROOT",
    "https://bhoonidhi-api.nrsc.gov.in/stac",
)


@dataclass
class StacSearchParams:
    collections: list[str]
    bbox: tuple[float, float, float, float]  # minx, miny, maxx, maxy
    datetime_range: str  # ISO interval e.g. 2024-11-01/2024-11-30
    cloud_cover_lt: float | None = 30.0
    limit: int = 10


class BhoonidhiStacClient:
    """Minimal STAC client — auth optional for open collections."""

    def __init__(self, token: str | None = None, root: str = BHOO_STAC_ROOT):
        self.root = root.rstrip("/")
        self.token = token or os.environ.get("BHOO_TOKEN")

    def _headers(self) -> dict[str, str]:
        h = {"Accept": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def search(self, params: StacSearchParams) -> dict[str, Any]:
        """POST /search — returns STAC ItemCollection JSON."""
        body: dict[str, Any] = {
            "collections": params.collections,
            "bbox": list(params.bbox),
            "datetime": params.datetime_range,
            "limit": params.limit,
        }
        if params.cloud_cover_lt is not None:
            body["query"] = {"eo:cloud_cover": {"lt": params.cloud_cover_lt}}
        url = f"{self.root}/search"
        resp = requests.post(url, json=body, headers=self._headers(), timeout=30)
        resp.raise_for_status()
        return resp.json()

    def example_rabi_wheat_malwa(self) -> StacSearchParams:
        """Example query: Malwa plateau rabi season Sentinel-2 L2A."""
        return StacSearchParams(
            collections=["sentinel2_l2a"],
            bbox=(74.5, 22.0, 76.5, 24.5),
            datetime_range="2024-11-01T00:00:00Z/2024-12-31T23:59:59Z",
            cloud_cover_lt=40.0,
        )


def demo_search_dry_run() -> dict[str, Any]:
    """
    Return a synthetic STAC response for offline demos.
    Live search requires Bhoonidhi registration.
    """
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "id": "demo-malwa-tile-001",
                "properties": {
                    "datetime": "2024-11-15T05:30:00Z",
                    "eo:cloud_cover": 12.4,
                    "bhoonidhi:collection": "sentinel2_l2a",
                    "crop_stage_hint": "vegetative",
                },
            }
        ],
        "note": "Synthetic — set BHOO_TOKEN for live NRSC STAC",
    }
