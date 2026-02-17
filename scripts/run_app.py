"""
CLI script to launch the Streamlit web application.

Usage:
    python scripts/run_app.py              # Default port 8501
    python scripts/run_app.py --port 8502  # Custom port
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch the PDF Search Engine web interface"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to run the application on (default: 8501)"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind to (default: localhost)"
    )

    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically"
    )

    return parser.parse_args()


def main():
    """Main entry point for launching the app."""
    args = parse_args()

    project_root = Path(__file__).parent.parent
    app_path = project_root / "src" / "gui" / "app.py"

    if not app_path.exists():
        print(f"Error: Application file not found: {app_path}")
        sys.exit(1)

    print("=" * 60)
    print("PDF Search Engine - Web Interface")
    print("=" * 60)
    print(f"Starting server on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(args.port),
        "--server.address", args.host,
    ]

    if args.no_browser:
        cmd.extend(["--server.headless", "true"])

    try:
        subprocess.run(cmd, cwd=str(project_root))
    except KeyboardInterrupt:
        print("\nShutting down...")
    except FileNotFoundError:
        print("Error: Streamlit not found. Install with: pip install streamlit")
        sys.exit(1)


if __name__ == "__main__":
    main()
