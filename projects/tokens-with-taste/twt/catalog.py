import json
import os
from importlib.resources import files

import httpx


LITELLM_SHA = "d034fb681ca296a0f082a224043eda62a2ce82d4"
PRICE_URL = f"https://raw.githubusercontent.com/BerriAI/litellm/{LITELLM_SHA}/model_prices_and_context_window.json"
OPENROUTER_URL = "https://openrouter.ai/api/v1/models"


def providers():
    return json.loads(files("twt").joinpath("providers.json").read_text())["providers"]


def missing(provider):
    return [key for key in [provider["key_env"], *provider.get("required_env", [])] if not os.environ.get(key)]


def base_url(provider):
    return provider["base_url"].replace("{account_id}", os.environ.get("CLOUDFLARE_ACCOUNT_ID", ""))


def headers(provider):
    return {"Authorization": f"Bearer {os.environ[provider['key_env']]}"}


def number(value):
    try:
        result = float(value)
        return result if result >= 0 and result < float("inf") else None
    except (TypeError, ValueError):
        return None


def normalize(provider, row, prices, router):
    model_id = row["id"]
    prefix = provider["map_prefix"]
    metadata = prices.get(f"{prefix}/{model_id}", {})
    if not metadata:
        candidate = prices.get(model_id, {})
        if candidate.get("litellm_provider") == prefix:
            metadata = candidate
    # Cross-reference exact IDs only: suffix matching can attach another model's capabilities.
    reference = row if provider["id"] == "openrouter" else router.get(model_id, {})
    params = reference.get("supported_parameters", [])
    mode = metadata.get("mode")
    architecture = reference.get("architecture", {})
    outputs = architecture.get("output_modalities", [])
    chat = "text" in outputs if outputs else mode == "chat"
    known = provider.get("models", {}).get(model_id, {})
    pricing = row.get("pricing", {}) if provider["id"] == "openrouter" else {}
    input_price = number(pricing.get("prompt"))
    output_price = number(pricing.get("completion"))
    if input_price is None:
        input_price = number(metadata.get("input_cost_per_token"))
    if output_price is None:
        output_price = number(metadata.get("output_cost_per_token"))
    free = known.get("free", False) or (provider["id"] == "openrouter" and
            number(pricing.get("prompt")) == 0 and number(pricing.get("completion")) == 0)
    if provider.get("free_prototyping"):
        free = True
    if "supported_parameters" in reference:
        tools = "tools" in params
        json_mode = "response_format" in params
        json_schema = "structured_outputs" in params
    else:
        tools = metadata.get("supports_function_calling") is True
        json_mode = json_schema = metadata.get("supports_response_schema") is True
    default_rpm = provider.get("rpm")
    if provider["id"] == "openrouter" and not model_id.endswith(":free"):
        default_rpm = None
    rpm = known.get("rpm", default_rpm)
    return {
        "provider": provider["id"], "id": model_id, "chat": chat,
        "tools": tools, "json": json_mode, "json_schema": json_schema,
        "context_length": reference.get("context_length") or metadata.get("max_input_tokens"),
        "input_price": 0.0 if free else input_price, "output_price": 0.0 if free else output_price,
        "free": free, "rpm": rpm,
        "daily": known.get("daily"), "synced_source": provider.get("models_url") or base_url(provider) + "/models",
        "metadata_source": OPENROUTER_URL if reference else PRICE_URL,
        "price_source": OPENROUTER_URL if provider["id"] == "openrouter" else PRICE_URL,
    }


async def sync(store, client=None, configuration=None):
    owned = client is None
    client = client or httpx.AsyncClient(timeout=60, follow_redirects=True)
    messages = []
    try:
        response = await client.get(PRICE_URL)
        response.raise_for_status()
        prices = response.json()
        response = await client.get(OPENROUTER_URL)
        response.raise_for_status()
        router_rows = response.json()["data"]
        router = {m["id"]: m for m in router_rows}
        messages.append(f"LiteLLM {LITELLM_SHA}: {len(prices)} entries")
        messages.append(f"OpenRouter public catalog: {len(router)} models (no key)")
        for provider in configuration or providers():
            absent = missing(provider)
            if provider["id"] == "openrouter":
                rows = router_rows
            elif absent and not provider.get("public_models"):
                messages.append(f"{provider['id']}: SKIPPED missing {', '.join(absent)}")
                continue
            elif provider.get("discovery_blocked"):
                messages.append(f"{provider['id']}: BLOCKED {provider['discovery_blocked']}")
                continue
            else:
                try:
                    response = await client.get(provider.get("models_url") or base_url(provider) + "/models",
                                                headers={} if absent else headers(provider))
                    response.raise_for_status()
                    rows = response.json()["data"]
                    if not isinstance(rows, list) or any(not isinstance(r, dict) or not isinstance(r.get("id"), str) for r in rows):
                        raise ValueError("Invalid model list")
                except (httpx.HTTPError, KeyError, ValueError):
                    messages.append(f"{provider['id']}: FAILED model listing; cached rows retained")
                    continue
            models = [normalize(provider, row, prices, router) for row in rows]
            store.replace_models(provider["id"], models)
            messages.append(f"{provider['id']}: synced {len(models)} models")
            if absent:
                messages.append(f"{provider['id']}: inference SKIPPED missing {', '.join(absent)}")
        return messages
    finally:
        if owned:
            await client.aclose()
