"""Shared pieces of the contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ContractModel(BaseModel):
    """Base for every contract document.

    Unknown fields are ignored on read (additive minors keep parsing) and
    never emitted; `by_alias` is not needed because field names are the
    camelCase wire names already."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True, ser_json_inf_nan="null")

    def to_json_dict(self) -> dict[str, Any]:
        """The document as JSON-ready data, `None` fields dropped except the
        ones a contract requires to be present (`error: null`)."""
        return self.model_dump(mode="json", exclude_none=True)


class OpenModel(BaseModel):
    """A block the forge copies without interpreting (`provenance`, `source`)."""

    model_config = ConfigDict(extra="allow")
