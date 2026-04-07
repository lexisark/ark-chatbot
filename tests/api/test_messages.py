"""Tests for message endpoints."""

import uuid


class TestSendMessage:
    async def test_send_message(self, client):
        create = await client.post("/api/chats", json={"title": "Msg test"})
        chat_id = create.json()["id"]

        resp = await client.post(f"/api/chats/{chat_id}/messages", json={
            "content": "Hello!",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["role"] == "assistant"
        assert len(data["content"]) > 0
        assert "id" in data

    async def test_send_message_stores_both(self, client):
        """Sending a message should store both user and assistant messages."""
        create = await client.post("/api/chats", json={"title": "Both test"})
        chat_id = create.json()["id"]

        await client.post(f"/api/chats/{chat_id}/messages", json={"content": "Hi"})

        history = await client.get(f"/api/chats/{chat_id}/messages")
        messages = history.json()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hi"
        assert messages[1]["role"] == "assistant"

    async def test_send_message_chat_not_found(self, client):
        resp = await client.post(f"/api/chats/{uuid.uuid4()}/messages", json={"content": "Hi"})
        assert resp.status_code == 404


class TestGetMessages:
    async def test_get_messages(self, client):
        create = await client.post("/api/chats", json={"title": "History"})
        chat_id = create.json()["id"]

        await client.post(f"/api/chats/{chat_id}/messages", json={"content": "msg 1"})
        await client.post(f"/api/chats/{chat_id}/messages", json={"content": "msg 2"})

        resp = await client.get(f"/api/chats/{chat_id}/messages")
        assert resp.status_code == 200
        messages = resp.json()
        # 2 user + 2 assistant = 4
        assert len(messages) == 4

    async def test_get_messages_with_limit(self, client):
        create = await client.post("/api/chats", json={"title": "Limit"})
        chat_id = create.json()["id"]

        for i in range(5):
            await client.post(f"/api/chats/{chat_id}/messages", json={"content": f"msg {i}"})

        resp = await client.get(f"/api/chats/{chat_id}/messages", params={"limit": 3})
        assert resp.status_code == 200
        assert len(resp.json()) == 3

    async def test_get_messages_chronological(self, client):
        create = await client.post("/api/chats", json={"title": "Order"})
        chat_id = create.json()["id"]

        await client.post(f"/api/chats/{chat_id}/messages", json={"content": "first"})
        await client.post(f"/api/chats/{chat_id}/messages", json={"content": "second"})

        resp = await client.get(f"/api/chats/{chat_id}/messages")
        messages = resp.json()
        assert messages[0]["content"] == "first"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["content"] == "second"
