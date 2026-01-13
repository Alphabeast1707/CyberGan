# ╔══════════════════════════════════════════════════════════════════╗
# ║  CyberGAN — Production Dockerfile                               ║
# ║  Multi-stage build: lean final image (~300MB vs ~2GB naive)     ║
# ║  Runs on any Linux server — Ubuntu, Debian, CentOS, etc.        ║
# ╚══════════════════════════════════════════════════════════════════╝

# ── Stage 1: Build dependencies ────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools (only needed at build time)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a separate prefix (for copying)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Production image ───────────────────────────────────────
FROM python:3.11-slim AS production

LABEL maintainer="CyberGAN"
LABEL description="CyberGAN — AI-Powered Server Security Agent"
LABEL version="1.0.0"

# Runtime system packages:
#   iptables    → Linux firewall (block IPs, ip6tables is bundled with it)
#   iproute2    → ip command (network ops)
#   net-tools   → netstat (connection monitoring)
#   procps      → ps, top (process monitoring)
#   libcap2-bin → setcap (drop root requirement)
#   curl        → healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    iptables \
    iproute2 \
    net-tools \
    procps \
    libcap2-bin \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

# Create a dedicated system user — drop privileges where possible
RUN groupadd -r cybergan \
    && useradd -r -g cybergan -d /opt/cybergan -s /sbin/nologin cybergan

WORKDIR /opt/cybergan

# Copy application source
COPY --chown=cybergan:cybergan . .

# Create runtime directories
RUN mkdir -p \
    /var/lib/cybergan/data \
    /var/log/cybergan \
    /opt/cybergan/checkpoints \
    /opt/cybergan/config/logs \
    && chown -R cybergan:cybergan \
        /var/lib/cybergan \
        /var/log/cybergan \
        /opt/cybergan

# Give Python NET_ADMIN + NET_RAW capabilities so iptables works
# without running the full container as root
RUN setcap cap_net_admin,cap_net_raw+eip /usr/local/bin/python3.11 2>/dev/null || true

# ── Volumes ────────────────────────────────────────────────────────
VOLUME ["/var/lib/cybergan", "/opt/cybergan/checkpoints", "/var/log/cybergan"]

# ── Network ────────────────────────────────────────────────────────
EXPOSE 8443

# ── Health check ───────────────────────────────────────────────────
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fs http://localhost:8443/api/status || exit 1

# ── Drop to cybergan user ──────────────────────────────────────────
USER cybergan

# ── Entrypoint ────────────────────────────────────────────────────
ENTRYPOINT ["python", "main.py"]

# Default: real monitoring agent + live dashboard
CMD ["run", "--mode", "hybrid", "--dashboard"]
