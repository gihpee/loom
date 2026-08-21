"""Ask one question and answer it: does the relay work from THIS machine?

    python -m loom_worker.p2p.doctor /ip4/203.0.113.7/tcp/47200/p2p/12D3KooW...

Diagnosing this through a running worker is guesswork, because everything that
can go wrong looks the same from outside — an unset address, a closed port, a
relay that refuses the reservation and a node that simply does not need one all
end as "no direct link, relaying". This starts a bare node with nothing but the
relay, waits, and says which of those it is.

Run it on the machine that is having trouble; the relay is only ever as
reachable as the network between the two.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

# A port of its own: the worker on this machine is probably using the usual one.
PROBE_PORT = 47399
# Reservations are normally taken within a second of connecting. The wait is
# long enough that a slow link is not mistaken for a broken one.
WAIT_S = 20.0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="loom-p2p-doctor",
        description="check that a circuit-relay is usable from this machine",
    )
    parser.add_argument(
        "relay",
        nargs="?",
        default=os.environ.get("LOOM_P2P_RELAY", ""),
        help="relay multiaddr (default: $LOOM_P2P_RELAY)",
    )
    parser.add_argument("--port", type=int, default=PROBE_PORT)
    parser.add_argument("--wait", type=float, default=WAIT_S)
    args = parser.parse_args(argv)

    if not args.relay:
        print("no relay address given, and LOOM_P2P_RELAY is empty.")
        print("Pass the multiaddr the relay printed at startup.")
        return 2
    try:
        from lattica import Lattica
    except Exception:
        print("the p2p stack is not installed here: pip install 'loom-worker[p2p]'")
        return 2

    print(f"relay:   {args.relay}")
    node = (
        Lattica.builder()
        .with_listen_addrs(
            [f"/ip4/0.0.0.0/tcp/{args.port}", f"/ip4/0.0.0.0/udp/{args.port}/quic-v1"]
        )
        # A throwaway identity: this probe must not disturb the worker's own.
        .with_key_path(tempfile.mkdtemp(prefix="loom-doctor-"))
        .with_dcutr(True)
        .with_autonat(True)
        .with_mdns(False)
        .with_upnp(False)
        .with_relay_servers([args.relay])
        .build()
    )
    print(f"me:      {node.peer_id()}")

    deadline = time.monotonic() + args.wait
    circuit = ""
    while time.monotonic() < deadline:
        for addr in node.get_visible_maddrs() or []:
            if "/p2p-circuit" in addr:
                circuit = addr
                break
        if circuit:
            break
        time.sleep(0.5)

    if not circuit:
        print(f"\nNO RESERVATION after {args.wait:.0f}s.")
        print("This machine could not get a slot on the relay. Usually one of:")
        print("  - the relay is not running, or not on that port")
        print("  - the port is closed on the way here (try: nc -vz <host> <port>)")
        print("  - the multiaddr's peer id belongs to an older relay identity")
        if "/quic" in args.relay:
            print("  - this is a QUIC address, and Lattica reserves only over")
            print("    TCP: it connects over QUIC and never sends RESERVE.")
            print("    Try the relay's /tcp/ address instead.")
        return 1

    # Held, or merely granted? A reservation that dies seconds later looks
    # identical at the moment it is taken, and that failure mode has already
    # happened once (a relay without ping: granted, then dropped, forever).
    print(f"\nreserved: {circuit}")
    print(f"holding for {args.wait:.0f}s to be sure it survives...")
    lost_at = None
    deadline = time.monotonic() + args.wait
    while time.monotonic() < deadline:
        time.sleep(1.0)
        if not any("/p2p-circuit" in a for a in node.get_visible_maddrs() or []):
            lost_at = time.monotonic()
            break
    if lost_at is not None:
        print("\nRESERVATION LOST while we watched.")
        print("The relay accepted us and then dropped the connection. If its")
        print("log shows nothing, check that it answers /ipfs/ping/1.0.0 —")
        print("Lattica drops a peer that does not.")
        return 1

    print("\nOK: the relay works from this machine and the reservation holds.")
    print("A worker here will be reachable through it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
