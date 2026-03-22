#!/usr/bin/env python3
"""
CyberGAN — AI-Powered Server Security Agent

CLI entry point for running the security agent, training the arena,
scanning for vulnerabilities, and managing the dashboard.

Usage:
    python main.py run                          # Start the security agent
    python main.py run --mode autonomous        # Full autonomous mode
    python main.py run --mode advisory          # Detection + alerts only
    python main.py train                        # Train Red vs Blue in arena
    python main.py train --epochs 500           # Custom training duration
    python main.py scan                         # One-shot security scan
    python main.py status                       # Show agent status
    python main.py dashboard                    # Launch web dashboard
"""

import argparse
import asyncio
import os
import signal
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="CyberGAN — AI-Powered Server Security Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  run        Start the CyberGAN security agent daemon
  train      Train Red vs Blue agents in the adversarial arena
  scan       Run a one-shot security scan of the system
  status     Show current security posture and agent status
  dashboard  Launch the web-based security dashboard
        """,
    )
    parser.add_argument("command", choices=["run", "train", "scan", "status", "dashboard"],
                        help="Command to execute")
    parser.add_argument("--config", default="config/default.yaml",
                        help="Path to configuration YAML file")
    parser.add_argument("--mode", choices=["advisory", "autonomous", "hybrid"],
                        help="Override agent mode")
    parser.add_argument("--epochs", type=int, help="Training epochs (for train command)")
    parser.add_argument("--dashboard", action="store_true",
                        help="Launch dashboard alongside training")
    parser.add_argument("--device", help="Override compute device (cpu/cuda)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Configure structured logging
    import logging
    import structlog
    log_level = logging.DEBUG if args.verbose else logging.INFO
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
    )

    # Load config
    from cybergan.config import CyberGANConfig

    try:
        config = CyberGANConfig.from_yaml(args.config)
    except FileNotFoundError:
        print(f"\033[93mConfig not found: {args.config}\033[0m")
        print("Using default configuration.")
        config = CyberGANConfig.default()

    # Apply overrides
    if args.mode:
        config.agent.mode = args.mode
    if args.device:
        config.training.device = args.device
        config.brain.device = args.device

    # Dispatch command
    if args.command == "run":
        _run_agent(config, with_dashboard=args.dashboard)
    elif args.command == "train":
        _run_training(config, args)
    elif args.command == "scan":
        _run_scan(config)
    elif args.command == "status":
        _show_status(config)
    elif args.command == "dashboard":
        _run_dashboard(config)


def _run_agent(config, with_dashboard: bool = False):
    """Start the CyberGAN security agent daemon."""
    from cybergan.agent import CyberGANAgent

    agent = CyberGANAgent(config)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    if with_dashboard:
        # ── Start dashboard + real agent together ──
        import threading
        import uvicorn
        from cybergan.dashboard.server import create_app, dashboard as dashboard_obj

        app = create_app()

        # Wire: agent → dashboard (real events flow in real time)
        async def _relay(payload: dict):
            await dashboard_obj.broadcast(payload)

        agent.set_dashboard(_relay)

        print(f"\n  {'═'*55}")
        print(f"  🛡️  CyberGAN — Real Monitoring Mode")
        print(f"  {'═'*55}")
        print(f"  Mode:      {config.agent.mode.upper()}")
        print(f"  Dashboard: http://127.0.0.1:{config.dashboard.port}")
        print(f"  Brain:     Heuristic (train first for RL)")
        print(f"  {'═'*55}\n")

        # Dashboard runs in a daemon thread; agent runs on the main loop
        def _run_uvicorn():
            uvicorn.run(
                app,
                host=config.dashboard.host,
                port=config.dashboard.port,
                log_level="warning",
            )

        dash_thread = threading.Thread(target=_run_uvicorn, daemon=True)
        dash_thread.start()
        import time as _time
        _time.sleep(1)  # Let dashboard bind before agent starts

    def shutdown_handler():
        loop.create_task(agent.stop())

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, shutdown_handler)
    except NotImplementedError:
        pass

    try:
        loop.run_until_complete(agent.start())
    except KeyboardInterrupt:
        loop.run_until_complete(agent.stop())
    finally:
        loop.close()


def _run_training(config, args):
    """Run arena training (Red vs Blue co-evolution)."""
    from rich.console import Console
    console = Console()

    try:
        from training.trainer import CyberGANTrainer

        config_path = args.config if os.path.exists(args.config) else None

        if config_path:
            trainer = CyberGANTrainer(config_path=config_path)
        else:
            # Use embedded config
            import tempfile, yaml
            tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
            yaml.dump(config.model_dump(), tmp, default_flow_style=False)
            tmp.close()
            trainer = CyberGANTrainer(config_path=tmp.name)

        if args.epochs:
            trainer.epochs = args.epochs

        if args.dashboard:
            _launch_with_dashboard(trainer, console)
        else:
            trainer.train()

    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        import traceback
        traceback.print_exc()


def _launch_with_dashboard(trainer, console):
    """Launch dashboard alongside training."""
    import threading
    import uvicorn
    from cybergan.dashboard.server import create_app, dashboard

    app = create_app(trainer)

    def on_epoch(data):
        dashboard.sync_broadcast(data)

    trainer.set_ws_callback(on_epoch)

    train_thread = threading.Thread(target=trainer.train, daemon=True)

    console.print("[cyan]Starting CyberGAN Dashboard at http://localhost:8443[/cyan]")
    console.print("[dim]Training runs in background. Open browser to view battle.[/dim]\n")

    train_thread.start()
    uvicorn.run(app, host="127.0.0.1", port=8443, log_level="warning")


def _run_scan(config):
    """Run a one-shot security scan."""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    console = Console()

    console.print(Panel(
        "[bold cyan]CyberGAN[/] — One-Shot Security Scan\n"
        "[dim]Scanning system for vulnerabilities and threats...[/dim]",
        border_style="cyan",
    ))

    import psutil

    # System info
    console.print("\n[bold]System Overview[/bold]")
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    console.print(f"  CPU:     {cpu}%")
    console.print(f"  Memory:  {mem.percent}% ({mem.used // 1024 // 1024} MB / {mem.total // 1024 // 1024} MB)")
    console.print(f"  Disk:    {disk.percent}% ({disk.used // 1024 // 1024 // 1024} GB / {disk.total // 1024 // 1024 // 1024} GB)")

    # Open ports
    console.print("\n[bold]Listening Ports[/bold]")
    table = Table(show_header=True)
    table.add_column("Port", style="cyan")
    table.add_column("Protocol")
    table.add_column("PID")
    table.add_column("Process")

    try:
        for conn in psutil.net_connections(kind="inet"):
            if conn.status == "LISTEN":
                try:
                    proc = psutil.Process(conn.pid) if conn.pid else None
                    table.add_row(
                        str(conn.laddr.port),
                        "TCP",
                        str(conn.pid or "-"),
                        proc.name() if proc else "-",
                    )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    table.add_row(str(conn.laddr.port), "TCP", str(conn.pid or "-"), "-")
    except (psutil.AccessDenied, PermissionError):
        console.print("[yellow]  ⚠ Insufficient permissions for port listing (run with sudo)[/yellow]")

    console.print(table)

    # Active connections summary
    try:
        connections = psutil.net_connections(kind="inet")
        established = sum(1 for c in connections if c.status == "ESTABLISHED")
        console.print(f"\n  Active connections: {established}")
        console.print(f"  Total connections:  {len(connections)}")
    except (psutil.AccessDenied, PermissionError):
        console.print("\n  [yellow]Connection info requires elevated permissions[/yellow]")

    # Process check
    console.print("\n[bold]Suspicious Process Check[/bold]")
    suspicious = []
    for proc in psutil.process_iter(["pid", "name", "cpu_percent", "cmdline"]):
        try:
            name = (proc.info["name"] or "").lower()
            if name in ["nc", "ncat", "netcat", "socat", "xmrig", "minerd", "cpuminer"]:
                suspicious.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if suspicious:
        console.print("[red]  ⚠ Suspicious processes found:[/red]")
        for p in suspicious:
            console.print(f"    PID {p['pid']}: {p['name']} (CPU: {p['cpu_percent']}%)")
    else:
        console.print("[green]  ✓ No suspicious processes detected[/green]")

    console.print(Panel("[green]Scan complete[/green]", border_style="green"))


def _show_status(config):
    """Show current agent status."""
    from rich.console import Console
    console = Console()
    console.print("[bold cyan]CyberGAN Agent Status[/bold cyan]")
    console.print(f"  Mode:    {config.agent.mode}")
    console.print(f"  Config:  Loaded")
    console.print(f"  Brain:   {'RL Model' if os.path.exists(config.brain.model_path) else 'Heuristic (no model)'}")

    # Check monitors
    monitors = [
        ("Log Monitor", config.perception.log_monitor.enabled),
        ("Network Monitor", config.perception.network_monitor.enabled),
        ("File Monitor", config.perception.file_monitor.enabled),
        ("Process Monitor", config.perception.process_monitor.enabled),
        ("Web Monitor", config.perception.web_monitor.enabled),
        ("System Metrics", config.perception.system_metrics.enabled),
    ]
    console.print("\n  [bold]Monitors:[/bold]")
    for name, enabled in monitors:
        status = "[green]✓[/green]" if enabled else "[red]✗[/red]"
        console.print(f"    {status} {name}")


def _run_dashboard(config):
    """Launch the web dashboard."""
    import uvicorn
    from cybergan.dashboard.server import create_app

    app = create_app()
    print(f"\n  CyberGAN Dashboard: http://{config.dashboard.host}:{config.dashboard.port}\n")
    uvicorn.run(app, host=config.dashboard.host, port=config.dashboard.port, log_level="info")


if __name__ == "__main__":
    main()
