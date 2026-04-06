"""
CyberGAN — Standalone Arena Trainer
Runs PPO training and broadcasts epoch results to the running dashboard.
The dashboard must already be running at ws://127.0.0.1:8443/ws

Usage:
    python train.py              # 300 epochs
    python train.py --epochs 500
    python train.py --epochs 100 --no-broadcast
"""

import sys
import os
import argparse
import asyncio
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="CyberGAN Arena Trainer")
    parser.add_argument("--epochs",       type=int, default=300)
    parser.add_argument("--config",       default="config/default.yaml")
    parser.add_argument("--no-broadcast", action="store_true",
                        help="Don't broadcast to dashboard")
    args = parser.parse_args()

    from training.trainer import CyberGANTrainer

    print(f"\n  {'═'*55}")
    print(f"  ⚔️  CyberGAN — Arena Training")
    print(f"  {'═'*55}")
    print(f"  Epochs:    {args.epochs}")
    print(f"  Config:    {args.config}")
    print(f"  Dashboard: http://127.0.0.1:8443")
    print(f"  {'═'*55}\n")

    trainer = CyberGANTrainer(config_path=args.config)
    trainer.epochs = args.epochs

    if not args.no_broadcast:
        # Non-blocking WebSocket broadcaster to existing dashboard
        import threading, queue
        _queue = queue.Queue(maxsize=200)

        def _ws_worker():
            """Background thread: drains queue and sends to dashboard WS."""
            try:
                import websocket   # websocket-client (sync)
                ws = websocket.create_connection(
                    "ws://127.0.0.1:8443/ws",
                    timeout=3,
                )
                print("  ✓ Connected to dashboard — training data streaming live")
                while True:
                    payload = _queue.get()
                    if payload is None:
                        break
                    try:
                        ws.send(json.dumps(payload))
                    except Exception:
                        # Try to reconnect
                        try:
                            ws = websocket.create_connection("ws://127.0.0.1:8443/ws", timeout=2)
                            ws.send(json.dumps(payload))
                        except Exception:
                            pass   # Dashboard may be busy — skip this frame
                ws.close()
            except Exception as e:
                print(f"  ⚠  Dashboard not reachable ({e}) — training without broadcast")
                # Drain the queue silently
                while True:
                    try:
                        _queue.get_nowait()
                    except queue.Empty:
                        import time; time.sleep(0.5)

        ws_thread = threading.Thread(target=_ws_worker, daemon=True)
        ws_thread.start()

        def _callback(payload: dict):
            try:
                _queue.put_nowait(payload)
            except queue.Full:
                pass  # Drop frame if dashboard is slow

        trainer.set_ws_callback(_callback)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n  Training interrupted. Saving current weights...")
        trainer._save_models()
        trainer._save_logs()

    # Signal broadcaster to stop
    if not args.no_broadcast:
        _queue.put(None)

    print(f"\n  ✓ Checkpoint: checkpoints/blue_production.pt")
    print(f"  ✓ Run the agent with RL brain:")
    print(f"     python main.py run --dashboard\n")


if __name__ == "__main__":
    # Check websocket-client is available
    try:
        import websocket
    except ImportError:
        print("Installing websocket-client...")
        os.system("pip install websocket-client -q")
    main()
