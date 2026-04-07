#!/usr/bin/env python3
import sys
import os

VENV_SITE_PACKAGES = '/home/dragan-slaveski/.openclaw/workspace/skills/xiaomi-home/venv/lib/python3.12/site-packages'
if VENV_SITE_PACKAGES not in sys.path:
    sys.path.insert(0, VENV_SITE_PACKAGES)

import json
import argparse
from micloud import MiCloud
from micloud.micloudexception import MiCloudException, MiCloudAccessDenied

XIAOMI_USERNAME = os.environ.get("XIAOMI_USERNAME")
XIAOMI_PASSWORD = os.environ.get("XIAOMI_PASSWORD")
XIAOMI_COUNTRY  = os.environ.get("XIAOMI_COUNTRY", "de")  # 'de' = Europe server


def get_client():
    if not XIAOMI_USERNAME or not XIAOMI_PASSWORD:
        print("❌ XIAOMI_USERNAME and XIAOMI_PASSWORD must be set in environment.")
        sys.exit(1)
    mc = MiCloud(XIAOMI_USERNAME, XIAOMI_PASSWORD)
    mc.default_server = XIAOMI_COUNTRY
    try:
        ok = mc.login()
    except MiCloudAccessDenied as e:
        print(f"❌ Access denied: {e}")
        sys.exit(1)
    if not ok:
        print("❌ Login failed. Check credentials.")
        sys.exit(1)
    return mc


def cmd_devices(args):
    mc = get_client()
    devices = mc.get_devices(country=XIAOMI_COUNTRY, raw=True)
    if not devices:
        print("No devices found.")
        return
    items = devices if isinstance(devices, list) else devices.get("result", {}).get("list", [])
    print(f"Found {len(items)} device(s):\n")
    for d in items:
        did   = d.get("did", "?")
        name  = d.get("name", "Unknown")
        model = d.get("model", "?")
        token = d.get("token", "(no token)")
        ip    = d.get("localip", "?")
        online = "🟢" if d.get("isOnline") else "🔴"
        print(f"{online} [{did}] {name}")
        print(f"     Model : {model}")
        print(f"     IP    : {ip}")
        print(f"     Token : {token}")
        print()


def cmd_raw(args):
    mc = get_client()
    devices = mc.get_devices(country=XIAOMI_COUNTRY, raw=True)
    print(json.dumps(devices, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Xiaomi Smart Home Cloud CLI")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("devices", help="List all devices with tokens and IPs")
    sub.add_parser("raw",     help="Dump raw device JSON from cloud")

    args = parser.parse_args()

    if args.command == "devices":
        cmd_devices(args)
    elif args.command == "raw":
        cmd_raw(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
