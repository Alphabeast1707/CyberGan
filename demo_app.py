"""
CyberGAN WAF — Demo App
A realistic FastAPI web app protected by CyberGAN WAF.
Run this, then run attack_tester.py to see the WAF in action.

Usage:
    python demo_app.py

Then in another terminal:
    python attack_tester.py

Watch the CyberGAN dashboard at http://127.0.0.1:8443
Every attack is detected in real time.
"""

import uvicorn
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

from cybergan_waf import CyberGANMiddleware

# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="CyberGAN Demo App",
    description="A protected web app. Try attacking it — the WAF will catch everything.",
    version="1.0.0",
)

# ── Add CyberGAN WAF ──────────────────────────────────────────
app.add_middleware(
    CyberGANMiddleware,
    mode="block",                              # ACTUALLY blocks malicious requests
    dashboard_url="ws://127.0.0.1:8443/ws",   # Real-time dashboard reporting
    rate_limit_per_minute=60,                  # 60 requests per minute per IP
    slack_webhook=None,                        # Set your Slack webhook here
    discord_webhook=None,                      # Set your Discord webhook here
    # NOTE: No whitelist_ips — localhost attacks ARE inspected (for demo purposes)
)

# ── Fake DB ───────────────────────────────────────────────────
USERS = {
    "admin": {"id": 1, "email": "admin@company.com", "role": "admin"},
    "alice": {"id": 2, "email": "alice@company.com", "role": "user"},
    "bob":   {"id": 3, "email": "bob@company.com",   "role": "user"},
}

POSTS = [
    {"id": 1, "title": "Hello World", "content": "Welcome to our blog."},
    {"id": 2, "title": "Security Tips", "content": "Always use HTTPS and strong passwords."},
    {"id": 3, "title": "New Features", "content": "We launched a new dashboard today."},
]

# ── Routes ────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>CyberGAN Demo App</title>
        <style>
            body { font-family: monospace; background: #0f0f0f; color: #e0e0e0; padding: 40px; }
            h1 { color: #22d3ee; }
            a { color: #22c55e; }
            .endpoint { background: #1a1a1a; border: 1px solid #2a2a2a; padding: 12px 16px; margin: 8px 0; border-radius: 6px; }
            .method { color: #f59e0b; font-weight: bold; }
            .badge { background: #e53e3e; color: white; padding: 2px 8px; border-radius: 3px; font-size: 11px; }
            .protected { color: #22c55e; font-size: 12px; }
        </style>
    </head>
    <body>
        <h1>🛡️ CyberGAN Demo App</h1>
        <p>This app is protected by CyberGAN WAF. Try attacking any endpoint — the WAF will block and report it.</p>
        <p class="protected">✅ CyberGAN WAF: ACTIVE (mode: block)</p>
        
        <h2>Endpoints (try attacking these):</h2>
        
        <div class="endpoint">
            <span class="method">GET</span> /users/search?name=alice<br>
            <small>Try: ?name=alice' UNION SELECT * FROM users--</small>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> /posts/1<br>
            <small>Try: /posts/1;DROP TABLE posts--</small>
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> /comments (body: {"text": "hello"})<br>
            <small>Try: {"text": "&lt;script&gt;alert(1)&lt;/script&gt;"}</small>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> /files?path=config.txt<br>
            <small>Try: ?path=../../../etc/passwd</small>
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> /ping (body: {"url": "https://example.com"})<br>
            <small>Try: {"url": "http://169.254.169.254/latest/meta-data"}</small>
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> /run (body: {"cmd": "echo hello"})<br>
            <small>Try: {"cmd": "echo hello; cat /etc/passwd"}</small>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> /api/stats — WAF runtime stats<br>
        </div>
        
        <p style="margin-top:30px; color:#555">
            Dashboard: <a href="http://127.0.0.1:8443">http://127.0.0.1:8443</a><br>
            Auto-test: <code>python attack_tester.py</code>
        </p>
    </body>
    </html>
    """


@app.get("/users/search")
async def search_users(name: str = "", role: str = ""):
    """Search users by name or role. Vulnerable to SQLi (WAF blocks it)."""
    results = [
        u for k, u in USERS.items()
        if name.lower() in k.lower() or (role and u["role"] == role)
    ]
    return {"users": results, "query": name}


@app.get("/posts/{post_id}")
async def get_post(post_id: str):
    """Get a post by ID. Vulnerable to injection in URL."""
    try:
        pid = int(post_id)
        post = next((p for p in POSTS if p["id"] == pid), None)
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        return post
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid post ID")


class CommentRequest(BaseModel):
    text: str
    author: Optional[str] = "anonymous"


@app.post("/comments")
async def post_comment(comment: CommentRequest):
    """Post a comment. Vulnerable to XSS (WAF blocks script tags)."""
    return {
        "status": "posted",
        "comment": {
            "text": comment.text,
            "author": comment.author,
            "timestamp": "2026-01-01T00:00:00Z",
        }
    }


@app.get("/files")
async def read_file(path: str = "readme.txt"):
    """Read a file. Vulnerable to LFI/path traversal (WAF blocks ../../)."""
    safe_files = {"readme.txt": "Welcome to CyberGAN!", "config.txt": "mode=production"}
    content = safe_files.get(path, f"File '{path}' not found")
    return {"path": path, "content": content}


class PingRequest(BaseModel):
    url: str


@app.post("/ping")
async def ping_url(req: PingRequest):
    """Ping a URL. Vulnerable to SSRF (WAF blocks 127.0.0.1, cloud metadata)."""
    # Simulated — doesn't actually make HTTP requests
    return {"status": "pong", "url": req.url, "reachable": True}


class RunRequest(BaseModel):
    cmd: str


@app.post("/run")
async def run_command(req: RunRequest):
    """Run a command. Vulnerable to RCE (WAF blocks shell injection)."""
    # Simulated — doesn't actually exec
    safe_cmds = {"echo hello": "hello", "date": "Wed May 27 2026", "whoami": "webapp"}
    result = safe_cmds.get(req.cmd.strip(), f"Command '{req.cmd}' executed")
    return {"output": result, "exit_code": 0}


@app.get("/login")
async def login_page(username: str = "", password: str = ""):
    """Login endpoint. Vulnerable to brute force + SQLi."""
    if username in USERS and password == "password123":
        return {"status": "success", "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."}
    return JSONResponse(status_code=401, content={"status": "error", "message": "Invalid credentials"})


@app.get("/api/stats")
async def waf_stats(request: Request):
    """Get WAF runtime statistics."""
    # Find WAF middleware instance
    for middleware in request.app.middleware_stack.__class__.__mro__:
        pass
    return {
        "app": "CyberGAN Demo",
        "waf": "active",
        "mode": "block",
        "endpoints": 7,
        "dashboard": "http://127.0.0.1:8443",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "service": "cybergan-demo"}


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═" * 55)
    print("  CyberGAN WAF — Demo Application")
    print("═" * 55)
    print("  App:       http://127.0.0.1:8000")
    print("  Dashboard: http://127.0.0.1:8443")
    print("  Attack it: python attack_tester.py")
    print("═" * 55 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
