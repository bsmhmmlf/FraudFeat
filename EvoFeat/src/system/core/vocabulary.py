from __future__ import annotations

import json
from functools import lru_cache

from system.paths import CONFIG_DIR


SCHEMA_PATH = CONFIG_DIR / "fraudfeat_schema.json"

_QUERY_ENTITY_ALIASES = {
    "WeChat Pay Account": "wechat_pay_account",
    "Merchant Account": "merchant_pay_account",
    "Merchant Pay Account": "merchant_pay_account",
    "Individual Customer": "individual_customer",
    "Natural Person": "natural_person",
    "Bank Card": "bank_card",
    "Phone": "phone_number",
    "Phone Number": "phone_number",
    "WeChat Delete Bill": "wechat_pay_bill_deletion_event",
    "WeChat Query Cardback": "wechat_pay_card_back_information_query_event",
}

_ENTITY_CATEGORIES = {
    "wechat_pay_account": "Account (account)",
    "merchant_pay_account": "Account (account)",
    "individual_customer": "Subject (subject)",
    "natural_person": "Subject (subject)",
    "bank_card": "Object (object)",
    "phone_number": "Object (object)",
    "wechat_pay_bill_deletion_event": "Operation behavior (operation_behavior)",
    "wechat_pay_card_back_information_query_event": "Operation behavior (operation_behavior)",
}


@lru_cache(maxsize=1)
def public_schema() -> dict:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    if schema.get("version") != 4:
        raise ValueError(f"Unsupported public vocabulary version: {schema.get('version')}")
    return schema


@lru_cache(maxsize=1)
def entity_definitions() -> dict:
    return {entity["entity_type"]: entity for entity in public_schema()["entities"]}


@lru_cache(maxsize=1)
def relation_definitions() -> dict:
    return {relation["en"]: relation for relation in public_schema()["relations"]}


@lru_cache(maxsize=1)
def _entity_aliases() -> dict:
    aliases = {entity.casefold(): entity for entity in entity_definitions()}
    for name, entity in _QUERY_ENTITY_ALIASES.items():
        aliases[name.casefold()] = entity
    return aliases


def canonical_entity(entity: str) -> str:
    value = str(entity or "").strip()
    return _entity_aliases().get(value.casefold(), value)


def is_public_entity(entity: str) -> bool:
    return canonical_entity(entity) in entity_definitions()


def entity_label(entity: str) -> str:
    return canonical_entity(entity) or "unknown"


def entity_category(entity: str) -> str:
    return _ENTITY_CATEGORIES.get(canonical_entity(entity), "")


def entity_prefix(entity: str) -> str:
    return {
        "Account (account)": "A",
        "Subject (subject)": "S",
        "Object (object)": "O",
        "Operation behavior (operation_behavior)": "B",
    }.get(entity_category(entity), "N")


def public_entity_fields(entity: str) -> frozenset:
    definition = entity_definitions().get(canonical_entity(entity)) or {}
    return frozenset(str(field["name"]) for field in definition.get("fields", []))


def public_relation_bindings() -> dict[str, tuple[str, ...]]:
    return {name: (name,) for name in relation_definitions()}


def _published_endpoints(relation: dict) -> tuple[str, str]:
    endpoints = (relation.get("join_fields") or [{}])[0]
    return (canonical_entity(endpoints.get("src") or ""),
            canonical_entity(endpoints.get("dst") or ""))


def public_relation(relation: str) -> dict:
    return relation_definitions().get(str(relation or "").strip()) or {}


def canonical_relation_name(relation: str) -> str:
    value = str(relation or "").strip()
    return str(public_relation(value).get("en") or value or "unknown")


def public_relation_endpoints(relation: str) -> tuple[str, str]:
    definition = public_relation(relation)
    return _published_endpoints(definition) if definition else ("", "")


def public_entity_view(node: dict) -> dict:
    result = dict(node)
    entity = canonical_entity(str(node.get("entity") or node.get("vtype")
                                  or node.get("entity_type") or node.get("type") or ""))
    if entity:
        result["entity"] = entity
        result["entity_type"] = entity_label(entity)
    return result
