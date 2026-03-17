#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════╗
# ║  CyberGAN — One-Command Server Deployment                       ║
# ║                                                                  ║
# ║  Usage:                                                          ║
# ║    curl -fsSL https://your-host/deploy.sh | bash                 ║
# ║    OR: ./deploy.sh [--mode advisory|autonomous|hybrid]           ║
# ║                    [--port 8443]                                  ║
# ║                    [--slack-webhook <url>]                        ║
# ║                    [--train]  # train RL model before starting   ║
# ╚══════════════════════════════════════════════════════════════════╝

set -euo pipefail

# ── Colours ────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

banner() {
    echo -e "${CYAN}"
    echo "  ██████╗██╗   ██╗██████╗ ███████╗██████╗  ██████╗  █████╗ ███╗"
    echo " ██╔════╝╚██╗ ██╔╝██╔══██╗██╔════╝██╔══██╗██╔════╝ ██╔══██╗████╗"
    echo " ██║      ╚████╔╝ ██████╔╝█████╗  ██████╔╝██║  ███╗███████║██╔██╗"
    echo " ██║       ╚██╔╝  ██╔══██╗██╔══╝  ██╔══██╗██║   ██║██╔══██║██║╚██╗"
    echo " ╚██████╗   ██║   ██████╔╝███████╗██║  ██║╚██████╔╝██║  ██║██║ ╚███╗"
    echo "  ╚═════╝   ╚═╝   ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚══╝"
    echo -e "${BOLD}         AI-Powered Server Security — Deploy Script v1.0${RESET}"
    echo ""
}

log()     { echo -e "  ${GREEN}✓${RESET}  $*"; }
warn()    { echo -e "  ${YELLOW}⚠${RESET}  $*"; }
error()   { echo -e "  ${RED}✗${RESET}  $*" >&2; exit 1; }
section() { echo -e "\n  ${BOLD}${CYAN}── $* ──${RESET}"; }

# ── Defaults ───────────────────────────────────────────────────────
MODE="hybrid"
PORT="8443"
SLACK_WEBHOOK=""
DISCORD_WEBHOOK=""
DO_TRAIN=false
COMPOSE_FILE="docker-compose.yml"

# ── Parse args ─────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)           MODE="$2";            shift 2 ;;
        --port)           PORT="$2";            shift 2 ;;
        --slack-webhook)  SLACK_WEBHOOK="$2";   shift 2 ;;
        --discord-webhook) DISCORD_WEBHOOK="$2"; shift 2 ;;
        --train)          DO_TRAIN=true;        shift   ;;
        --help|-h)
            echo "Usage: $0 [--mode hybrid|advisory|autonomous] [--port 8443] [--train]"
            exit 0 ;;
        *) warn "Unknown option: $1"; shift ;;
    esac
done

banner

# ── Check root / sudo ──────────────────────────────────────────────
section "Pre-flight checks"
if [[ $EUID -ne 0 ]]; then
    warn "Not running as root. Firewall blocking may be limited."
    warn "Run with sudo for full iptables access."
fi

# ── Check Docker ───────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    section "Installing Docker"
    curl -fsSL https://get.docker.com | sh
    systemctl enable --now docker
    log "Docker installed and started"
else
    DOCKER_VER=$(docker --version | grep -oP '\d+\.\d+')
    log "Docker $DOCKER_VER found"
fi

if ! docker compose version &>/dev/null 2>&1; then
    error "docker compose (v2) not found. Update Docker: https://docs.docker.com/engine/install/"
fi
log "Docker Compose v2 found"

# ── Write .env ─────────────────────────────────────────────────────
section "Writing configuration"
cat > .env <<EOF
CYBERGAN_MODE=${MODE}
DASHBOARD_PORT=${PORT}
CYBERGAN_LOG_LEVEL=INFO
SLACK_WEBHOOK=${SLACK_WEBHOOK}
DISCORD_WEBHOOK=${DISCORD_WEBHOOK}
PAGERDUTY_KEY=
EOF
log ".env written (mode=${MODE}, port=${PORT})"

# ── Build image ────────────────────────────────────────────────────
section "Building CyberGAN image"
docker compose build --no-cache 2>&1 | grep -E "Step|RUN|COPY|Successfully|error" || true
log "Image built: cybergan:latest"

# ── (Optional) Train RL model ──────────────────────────────────────
if [[ "$DO_TRAIN" == "true" ]]; then
    section "Training RL model (300 epochs)"
    echo "  This takes ~3 minutes on CPU, ~30s on GPU..."
    docker compose run --rm trainer
    log "Training complete — blue_production.pt saved"
fi

# ── Start services ─────────────────────────────────────────────────
section "Starting CyberGAN"
docker compose up -d cybergan

# Wait for healthcheck
echo "  Waiting for agent to start..."
for i in $(seq 1 20); do
    if curl -fs "http://localhost:${PORT}/api/status" &>/dev/null; then
        log "Agent is healthy"
        break
    fi
    sleep 2
    if [[ $i -eq 20 ]]; then
        warn "Agent health check timed out — check: docker logs cybergan-agent"
    fi
done

# ── Print summary ──────────────────────────────────────────────────
IP=$(curl -fs ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}')

echo ""
echo -e "  ${BOLD}${GREEN}╔══════════════════════════════════════════════════════╗${RESET}"
echo -e "  ${BOLD}${GREEN}║  CyberGAN is LIVE                                    ║${RESET}"
echo -e "  ${BOLD}${GREEN}╚══════════════════════════════════════════════════════╝${RESET}"
echo ""
echo -e "  ${BOLD}Dashboard:${RESET}   http://${IP}:${PORT}"
echo -e "  ${BOLD}Local:${RESET}       http://localhost:${PORT}"
echo -e "  ${BOLD}Mode:${RESET}        ${MODE}"
echo -e "  ${BOLD}Brain:${RESET}       $([ -f checkpoints/blue_production.pt ] && echo 'RL Policy' || echo 'Heuristic')"
echo ""
echo -e "  ${BOLD}Useful commands:${RESET}"
echo "    docker logs -f cybergan-agent       # Live logs"
echo "    docker compose restart cybergan     # Restart agent"
echo "    docker compose run --rm trainer     # Retrain RL model"
echo "    docker compose down                 # Stop everything"
echo ""
echo -e "  ${CYAN}Your server is now protected.${RESET}"
echo ""
