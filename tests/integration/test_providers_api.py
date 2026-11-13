"""tests/integration/test_providers_api.py"""

from __future__ import annotations


async def test_list_providers(env):
    """GET /providers returns all providers (read-only endpoint for any user)."""
    c, _, auth = env
    r = await c.post(
        "/api/admin/providers",
        headers=auth,
        json={"name": "openai", "kind": "openai", "api_key": "k"},
    )
    assert r.status_code == 201

    r = await c.get("/api/providers", headers=auth)
    assert r.status_code == 200
    assert any(p["name"] == "openai" for p in r.json())


async def test_admin_creates_provider(env):
    """POST /admin/providers creates a provider."""
    c, _, auth = env
    r = await c.post(
        "/api/admin/providers",
        headers=auth,
        json={"name": "openai", "kind": "openai", "api_key": "k"},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["name"] == "openai"
    assert "user_id" not in body


async def test_duplicate_provider_name_is_conflict(env):
    """POST /admin/providers with a taken name is 409, not an unhandled 500."""
    c, _, auth = env
    body = {"name": "taken-name", "kind": "openai", "api_key": "k"}
    created = await c.post("/api/admin/providers", headers=auth, json=body)
    assert created.status_code == 201

    again = await c.post("/api/admin/providers", headers=auth, json=body)
    assert again.status_code == 409
    assert again.json()["error"]["code"] == "PROVIDER_NAME_TAKEN"


async def test_regular_user_cannot_create_provider(env):
    """POST /admin/providers is admin-only."""
    c, _, admin_auth = env
    await c.post(
        "/api/users",
        headers=admin_auth,
        json={"username": "regular", "password": "TestPass12", "role": "user"},
    )
    tok = (
        await c.post("/api/auth/login", json={"username": "regular", "password": "TestPass12"})
    ).json()["access_token"]
    user_auth = {"Authorization": f"Bearer {tok}"}

    r = await c.post(
        "/api/admin/providers",
        headers=user_auth,
        json={"name": "should-fail", "kind": "openai"},
    )
    assert r.status_code == 403


async def test_admin_delete_provider(env):
    """DELETE /admin/providers/{id} removes the provider."""
    c, _, auth = env
    r = await c.post(
        "/api/admin/providers",
        headers=auth,
        json={"name": "to-delete", "kind": "openai"},
    )
    assert r.status_code == 201
    pid = r.json()["id"]

    r = await c.delete(f"/api/admin/providers/{pid}", headers=auth)
    assert r.status_code in (200, 204)

    r = await c.get("/api/admin/providers", headers=auth)
    ids = [p["id"] for p in r.json()]
    assert pid not in ids


async def test_admin_cannot_delete_local_runtime_provider(env):
    """DELETE /admin/providers/{id} keeps ONNX / Ollama local rows."""
    c, _, auth = env
    r = await c.post(
        "/api/admin/providers",
        headers=auth,
        json={"name": "ONNX (Local)", "kind": "openai", "api_key": "onnx"},
    )
    assert r.status_code == 201
    pid = r.json()["id"]

    r = await c.delete(f"/api/admin/providers/{pid}", headers=auth)
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "PROVIDER_LOCAL_PROTECTED"

    r = await c.get("/api/admin/providers", headers=auth)
    ids = [p["id"] for p in r.json()]
    assert pid in ids


async def test_admin_can_delete_custom_provider_on_local_url(env):
    """Custom OpenAI-compat rows stay deletable even on Ollama-like URLs."""
    c, _, auth = env
    for name, base_url in (
        ("my-local-llm", "http://127.0.0.1:8000/v1"),
        ("new-api", "http://192.168.1.10:3000/v1"),
        ("company-ollama", "http://127.0.0.1:11434"),
        ("docker-ollama", "http://ollama:11434"),
    ):
        r = await c.post(
            "/api/admin/providers",
            headers=auth,
            json={
                "name": name,
                "kind": "openai",
                "api_key": "sk-local",
                "base_url": base_url,
                "models": [{"id": "custom-qwen", "name": "custom-qwen", "enabled": True}],
            },
        )
        assert r.status_code == 201, name
        pid = r.json()["id"]

        r = await c.delete(f"/api/admin/providers/{pid}", headers=auth)
        assert r.status_code in (200, 204), name

        r = await c.get("/api/admin/providers", headers=auth)
        ids = [p["id"] for p in r.json()]
        assert pid not in ids, name


async def test_admin_cannot_delete_ollama_runtime_by_identity(env):
    """Onboard Ollama is protected by placeholder key / exact name, not URL."""
    c, _, auth = env
    r = await c.post(
        "/api/admin/providers",
        headers=auth,
        json={
            "name": "Ollama (Local)",
            "kind": "openai",
            "api_key": "ollama",
            "base_url": "http://127.0.0.1:11434",
        },
    )
    assert r.status_code == 201
    pid = r.json()["id"]

    r = await c.delete(f"/api/admin/providers/{pid}", headers=auth)
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "PROVIDER_LOCAL_PROTECTED"


async def test_admin_stream_usage_defaults_and_patch(env):
    """Custom providers default stream_usage off; presets stay on until patched."""
    c, _, auth = env
    custom = await c.post(
        "/api/admin/providers",
        headers=auth,
        json={"name": "mindie-arm", "kind": "openai", "api_key": "k"},
    )
    assert custom.status_code == 201
    assert custom.json()["stream_usage"] is False

    preset = await c.post(
        "/api/admin/providers",
        headers=auth,
        json={"name": "DeepSeek", "kind": "openai", "api_key": "k"},
    )
    assert preset.status_code == 201
    assert preset.json()["stream_usage"] is True

    patched = await c.patch(
        f"/api/admin/providers/{preset.json()['id']}",
        headers=auth,
        json={"stream_usage": False},
    )
    assert patched.status_code == 200
    assert patched.json()["stream_usage"] is False


async def test_set_active_model_rejects_unavailable_ref(env):
    """PUT /providers/active-model refuses a ref that cannot serve a request.

    Without the check the setting is stored, the UI shows the switch, and every
    agent silently falls back to the first usable model.
    """
    c, _, auth = env
    created = await c.post(
        "/api/admin/providers",
        headers=auth,
        json={
            "name": "acme",
            "kind": "openai",
            "base_url": "https://acme.example/v1",
            "api_key": "k",
            "models": [
                {
                    "id": "acme-1",
                    "name": "Acme 1",
                    "enabled": True,
                    "input": ["text"],
                    "thinking": False,
                }
            ],
        },
    )
    assert created.status_code == 201

    ok = await c.put(
        "/api/providers/active-model",
        headers=auth,
        json={"provider_name": "acme", "model": "acme-1"},
    )
    assert ok.status_code == 200
    assert ok.json() == {"provider_name": "acme", "model": "acme-1"}

    for provider_name, model in (("acme", "acme-2"), ("ghost", "acme-1")):
        rejected = await c.put(
            "/api/providers/active-model",
            headers=auth,
            json={"provider_name": provider_name, "model": model},
        )
        assert rejected.status_code == 400, (provider_name, model)
        assert rejected.json()["error"]["code"] == "MODEL_NOT_AVAILABLE"

    # The failed attempts must not have overwritten the working setting.
    current = await c.get("/api/providers/active-model", headers=auth)
    assert current.json() == {"provider_name": "acme", "model": "acme-1"}
