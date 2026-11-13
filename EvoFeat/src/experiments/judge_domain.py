"""EvoFeat access to the benchmark's schema-derived judge domain brief."""

from __future__ import annotations

from functools import lru_cache

from system.core.vocabulary import public_schema
from system.execution.parquet_backend import fraudfeat_module


@lru_cache(maxsize=1)
def domain_brief() -> str:
    protocol = fraudfeat_module("interpretability")
    return protocol.build_domain_brief(public_schema())


def domain_sha1() -> str:
    return fraudfeat_module("interpretability").domain_sha1(domain_brief())


__all__ = ["domain_brief", "domain_sha1"]
