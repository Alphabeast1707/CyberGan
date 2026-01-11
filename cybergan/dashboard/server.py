"""
CyberGAN — Dashboard: FastAPI + WebSocket Server
Serves the real-time security operations dashboard.
"""

from __future__ import annotations

import json
import asyncio
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import os


class DashboardServer:
    """Manages WebSocket connections and event broadcasting."""

    def __init__(self):
        self.connections: list[WebSocket] = []
        self.event_buffer: list[dict] = []
        self.buffer_size = 100

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connections.append(websocket)
        # Send buffered events to new connections
        for event in self.event_buffer[-20:]:
            try:
                await websocket.send_json(event)
            except Exception:
                break

    def disconnect(self, websocket: WebSocket):
        if websocket in self.connections:
            self.connections.remove(websocket)

    async def broadcast(self, data: dict):
        """Send data to all connected WebSocket clients."""
        self.event_buffer.append(data)
        if len(self.event_buffer) > self.buffer_size:
            self.event_buffer = self.event_buffer[-self.buffer_size:]

        dead = []
        for ws in self.connections:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    def sync_broadcast(self, data: dict):
        """Synchronous wrapper for broadcast (called from training thread)."""
        self.event_buffer.append(data)
        if len(self.event_buffer) > self.buffer_size:
            self.event_buffer = self.event_buffer[-self.buffer_size:]


# Global dashboard state
dashboard = DashboardServer()
agent_ref = None
trainer_ref = None


def create_app(agent=None, trainer=None) -> FastAPI:
    """Create the FastAPI application."""
    global agent_ref, trainer_ref
    agent_ref = agent
    trainer_ref = trainer

    app = FastAPI(title="CyberGAN Security Dashboard")

    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        index_path = os.path.join(static_dir, "index.html")
        if os.path.exists(index_path):
            with open(index_path) as f:
                return HTMLResponse(content=f.read())
        return HTMLResponse(content="<h1>CyberGAN Dashboard</h1><p>Static files not found.</p>")

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await dashboard.connect(ws)
        try:
            while True:
                data = await ws.receive_text()
                if data == "ping":
                    await ws.send_json({"type": "pong"})
                elif data == "get_state":
                    state = _get_state()
                    await ws.send_json({"type": "full_state", "data": state})
                else:
                    # Relay arbitrary JSON events (threat simulator, agent, etc.)
                    try:
                        payload = json.loads(data)
                        await dashboard.broadcast(payload)
                    except (json.JSONDecodeError, Exception):
                        pass
        except (WebSocketDisconnect, RuntimeError):
            dashboard.disconnect(ws)

    @app.get("/api/status")
    async def get_status():
        if agent_ref:
            return JSONResponse(agent_ref.get_status())
        return JSONResponse({"status": "no agent attached"})

    @app.get("/api/state")
    async def get_state():
        return JSONResponse(_get_state())

    @app.get("/api/alerts")
    async def get_alerts():
        if agent_ref and hasattr(agent_ref, 'alerter'):
            return JSONResponse({"alerts": agent_ref.alerter.get_history()})
        return JSONResponse({"alerts": []})

    @app.get("/api/threats")
    async def get_threats():
        if agent_ref and hasattr(agent_ref, 'threat_classifier'):
            return JSONResponse({"threats": agent_ref.threat_classifier.get_active_threats()})
        return JSONResponse({"threats": []})

    return app


def _get_state() -> dict:
    """Get combined state from agent and/or trainer."""
    state = {}
    if agent_ref:
        state["agent"] = agent_ref.get_status()
    if trainer_ref:
        state["training"] = trainer_ref.get_state()
    return state
