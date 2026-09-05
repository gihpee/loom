"""This node's direct path to its neighbours.

Brought up when the rendezvous becomes known, which is NOT process start: the
address to bootstrap against arrives in the registration ack. So the link table
exists from the beginning and relays everything, and the p2p node is attached
to it later — everything downstream holds the same object and is never rewired.

Nothing here is fatal. A node that cannot join the peer network is slower, not
broken, and dropping a working GPU out of the pool over a networking nicety
would be the wrong trade.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable, List, Optional

from looma_agent.p2p.links import LinkTable
from looma_agent.p2p.peer import (
    DEFAULT_P2P_PORT,
    PeerNode,
    behind_container_nat,
    lattica_available,
)
from looma_agent.proto import agent_pb2

logger = logging.getLogger("looma_agent.p2p")

# The numbers this samples change on the scale of a network path settling, not
# of a token. Slow on purpose.
SAMPLE_INTERVAL_S = 15.0


def _env_relays() -> List[str]:
    return [a.strip() for a in os.environ.get("LOOMA_P2P_RELAY", "").split(",") if a.strip()]


def _enabled() -> bool:
    return os.environ.get("LOOMA_P2P", "1").strip().lower() not in ("0", "false", "no")


class PeerLayer:
    def __init__(
        self,
        *,
        on_message: Optional[Callable[[bytes], None]] = None,
        port: Optional[int] = None,
        key_dir: str = "",
    ) -> None:
        self.links = LinkTable()
        self.node: Optional[PeerNode] = None
        self.identity = None
        # Delivery of an inbound direct message. None until transport lands
        # (docs/AGENT_PLAN.md phase 7): a node with nothing to deliver to still
        # benefits from joining, because its neighbours can then reach IT.
        self._on_message = on_message or (lambda _payload: None)
        # Everything the heartbeat reports about reachability, sampled on the
        # sampler thread and read from here. Never measured in the heartbeat:
        # asking the p2p stack means entering its runtime, and a heartbeat that
        # does that stops arriving the moment the runtime is busy.
        self._visible: List[str] = []
        # This node's own distance to the relay. Measured directly rather than
        # read off the link table, because neighbours need it BEFORE any link
        # exists — a zero there means they cannot judge their side of the path.
        self._relay_rtt_ms: float = 0.0
        # Explicit so several agents can run in one process during tests: two
        # nodes sharing a key directory interfere, and a closed node does not
        # give its port back instantly.
        self._port = port
        self._key_dir = key_dir
        self._sampler: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ setup
    def on_rendezvous(self, addrs: List[str], relays: Optional[List[str]] = None) -> None:
        """Адреса точки встречи из ответа на регистрацию.

        Зовётся при КАЖДОЙ регистрации, а не только при первой: узел, не
        вошедший в сеть с первого раза, получает здесь второй шанс. Раньше
        отсюда молча выходили, если p2p-узел уже создан, — и узел, стартовавший
        в неудачный момент, оставался вне сети до перезапуска процесса. Чинилось
        это релизом на весь парк, а не перерегистрацией.
        """
        if not _enabled():
            return
        addrs = [a for a in (addrs or []) if a.strip()]
        if self.node is not None:
            self._rejoin(addrs, relays)
            return
        if not addrs:
            logger.info("the orchestrator offers no rendezvous; messages go through it")
            return
        if not lattica_available():
            logger.info(
                "no p2p stack installed; messages go through the orchestrator. "
                "Install the extra to let this node talk to its neighbours "
                "directly: pip install 'looma-agent[p2p]'"
            )
            return
        self._bring_up(addrs, relays)

    def _rejoin(self, addrs: List[str], relays: Optional[List[str]]) -> None:
        """Ещё раз войти в сеть — пересобрав узел.

        Адреса точки встречи вшиваются в узел при сборке, поэтому «попробовать
        снова» иначе не выражается. Делается это ТОЛЬКО когда узел ни с кем не
        соединён: пересобирать работающий значит рвать живые туннели ради
        задачи, которая уже решена.
        """
        if not addrs or self.node is None or self.node.connected_peers():
            return
        logger.info("p2p node is in the network with nobody; rebuilding it against %s",
                    addrs[0])
        try:
            self.node.close()
        except Exception:
            logger.warning("the old p2p node did not close cleanly", exc_info=True)
        # Сначала None: цикл выборки состояния смотрит на это поле и так
        # завершается, не оставляя второго такого же рядом с новым.
        self.node = None
        self.identity = None
        self._bring_up(addrs, relays)

    def _bring_up(self, addrs: List[str], relays: Optional[List[str]]) -> None:
        # The rendezvous is a DHT entry point and nothing else. Pointing the
        # relay client at it leaves the node waiting for a reservation that
        # never comes (measured: 15s, zero peers). A real circuit-relay server
        # is a separate process; see relay/relay.mjs.
        relay_addrs = [a for a in (relays or []) if a.strip()] or _env_relays()
        options = {"bootstraps": addrs, "relay_servers": relay_addrs}
        if self._port is not None:
            options["port"] = self._port
        if self._key_dir:
            options["key_dir"] = self._key_dir
        node = PeerNode(**options)
        try:
            self.identity = node.start(on_message=self._on_message)
        except Exception:
            logger.warning("the p2p node did not start; relaying everything", exc_info=True)
            return
        self.node = node
        self._start_sampler()
        self.links.attach(
            send_direct=node.send_nowait,
            dial=node.warm,
            rtt=node.rtt_ms,
            relay_rtt=node.relay_rtt_ms,
        )
        self._report_reachability(relay_addrs)

    def _report_reachability(self, relay_addrs: List[str]) -> None:
        """Say plainly which of the several silent failures this node is in.

        They look identical from the outside — node up, peer id reported,
        nobody can reach it — and have nothing in common. Naming which one it
        is saves the whole investigation.
        """
        identity = self.identity
        if identity is None:
            return
        if identity.symmetric_nat:
            logger.warning(
                "this node is behind a symmetric NAT: peers cannot open a direct "
                "link to it and will relay"
            )
            return
        if any("/p2p-circuit" in a for a in identity.visible_addrs):
            # A reservation is held: nothing can dial this node directly, but
            # peers reach it through the relay and can try to punch through
            # from there. This is the state the relay exists to produce.
            logger.info("this node is reachable through the relay: %s", identity.visible_addrs[0])
            return
        if identity.visible_addrs:
            return
        port = self.node.port if self.node else DEFAULT_P2P_PORT
        if behind_container_nat():
            logger.warning(
                "this agent runs on a Docker bridge network, so no peer can ever "
                "open a direct link to it: the container's port is not the host's, "
                "and outgoing packets are translated again on the way out. Hole "
                "punching cannot work from here. Restart with --network host "
                "(and open port %d) to make the direct path possible",
                port,
            )
        elif relay_addrs:
            logger.warning(
                "no address of this node is reachable from outside, and the relay "
                "at %s gave it no reservation either. Check that the relay is "
                "running and its port is open from here",
                relay_addrs[0],
            )
        else:
            logger.warning(
                "no address of this node is reachable from outside (AutoNAT "
                "confirmed none) and no relay was offered. Peers will relay TO it "
                "through the orchestrator; it can still send directly. Forward "
                "port %d (TCP and UDP), or run a relay (docs/P2P_RELAY.md)",
                port,
            )

    def _start_sampler(self) -> None:
        if self._sampler is not None and self._sampler.is_alive():
            return          # пересборка узла не повод заводить вторую такую же

        def sample() -> None:
            while self.node is not None:
                try:
                    self._visible = self.node.visible_addrs()
                    self._relay_rtt_ms = self.node.relay_rtt_ms() or 0.0
                    # A circuit address is not reachability: it means "through
                    # the relay", which is the relay path under another name.
                    self.links.set_self_reachable(
                        any("/p2p-circuit" not in a for a in self._visible)
                    )
                    self.links.refresh()
                except Exception:
                    logger.debug("sampling the p2p state failed", exc_info=True)
                time.sleep(SAMPLE_INTERVAL_S)

        self._sampler = threading.Thread(target=sample, name="looma-p2p-sampler",
                                         daemon=True)
        self._sampler.start()

    # ----------------------------------------------------------------- report
    def status(self) -> agent_pb2.PeerStatus:
        """What this node reports about its p2p state on every heartbeat.

        Reachability is re-read rather than taken from the identity captured at
        startup: it is not a constant. A relay reservation can arrive a second
        after the node joins, and AutoNAT needs a few probes before it confirms
        anything. Reporting the startup snapshot forever showed nodes as
        unreachable long after they had stopped being so.

        Counters are reported even with no identity at all: a node with no p2p
        stack relays everything, and that is exactly what the orchestrator
        needs to see.
        """
        stats = self.links.snapshot()
        identity = self.identity
        return agent_pb2.PeerStatus(
            peer_id=identity.peer_id if identity else "",
            symmetric_nat=bool(identity.symmetric_nat) if identity else False,
            direct=stats["direct"],
            relayed=stats["relay"],
            fallbacks=stats["fallbacks"],
            direct_share=stats["direct_share"],
            visible_addrs=self._visible or (identity.visible_addrs if identity else []),
            link_rtt_ms=stats["link_rtt_ms"],
            relay_rtt_ms=self._relay_rtt_ms or stats["relay_rtt_ms"],
        )

    def identity_message(self):
        """How peers can reach this node, for the registration. Empty until up."""
        identity = self.identity
        if identity is None:
            return None
        return identity
