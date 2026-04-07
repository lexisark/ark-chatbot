"""Tests for chat CRUD endpoints."""

import uuid


class TestCreateChat:
    async def test_create_chat(self, client):
        resp = await client.post("/api/chats", json={
            "title": "Test Chat",
            "system_prompt": "Be helpful.",
            "scope_id": "user-1",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["title"] == "Test Chat"
        assert data["system_prompt"] == "Be helpful."
        assert data["scope_id"] == "user-1"
        assert "id" in data

    async def test_create_chat_minimal(self, client):
        resp = await client.post("/api/chats", json={})
        assert resp.status_code == 201
        data = resp.json()
        assert "id" in data

    async def test_create_chat_with_metadata(self, client):
        resp = await client.post("/api/chats", json={
            "title": "Meta Chat",
            "metadata": {"key": "value"},
        })
        assert resp.status_code == 201
        assert resp.json()["metadata"]["key"] == "value"


class TestGetChat:
    async def test_get_chat(self, client):
        create = await client.post("/api/chats", json={"title": "Find me"})
        chat_id = create.json()["id"]

        resp = await client.get(f"/api/chats/{chat_id}")
        assert resp.status_code == 200
        assert resp.json()["title"] == "Find me"

    async def test_get_chat_not_found(self, client):
        resp = await client.get(f"/api/chats/{uuid.uuid4()}")
        assert resp.status_code == 404


class TestListChats:
    async def test_list_chats(self, client):
        await client.post("/api/chats", json={"title": "A"})
        await client.post("/api/chats", json={"title": "B"})

        resp = await client.get("/api/chats")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 2

    async def test_list_chats_with_scope(self, client):
        await client.post("/api/chats", json={"title": "S1", "scope_id": "scope-x"})
        await client.post("/api/chats", json={"title": "S2", "scope_id": "scope-y"})

        resp = await client.get("/api/chats", params={"scope_id": "scope-x"})
        assert resp.status_code == 200
        data = resp.json()
        assert all(c["scope_id"] == "scope-x" for c in data)

    async def test_list_chats_pagination(self, client):
        for i in range(5):
            await client.post("/api/chats", json={"title": f"Page {i}"})

        resp = await client.get("/api/chats", params={"limit": 2, "offset": 0})
        assert len(resp.json()) == 2


class TestUpdateChat:
    async def test_update_chat(self, client):
        create = await client.post("/api/chats", json={"title": "Old"})
        chat_id = create.json()["id"]

        resp = await client.patch(f"/api/chats/{chat_id}", json={
            "title": "New Title",
            "system_prompt": "Updated prompt",
        })
        assert resp.status_code == 200
        assert resp.json()["title"] == "New Title"
        assert resp.json()["system_prompt"] == "Updated prompt"

    async def test_update_chat_not_found(self, client):
        resp = await client.patch(f"/api/chats/{uuid.uuid4()}", json={"title": "Nope"})
        assert resp.status_code == 404


class TestDeleteChat:
    async def test_delete_chat(self, client):
        create = await client.post("/api/chats", json={"title": "Delete me"})
        chat_id = create.json()["id"]

        resp = await client.delete(f"/api/chats/{chat_id}")
        assert resp.status_code == 204

        get_resp = await client.get(f"/api/chats/{chat_id}")
        assert get_resp.status_code == 404

    async def test_delete_chat_not_found(self, client):
        resp = await client.delete(f"/api/chats/{uuid.uuid4()}")
        assert resp.status_code == 404
