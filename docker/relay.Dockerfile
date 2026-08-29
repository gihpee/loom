# libp2p circuit-relay v2 server for Loom (see docs/P2P_RELAY.md).
#
# Its own image and its own process because the workers' p2p stack, Lattica,
# is a relay CLIENT only: it announces /libp2p/circuit/relay/0.2.0/stop and
# never the /hop half a server offers. No configuration changes that, so the
# server has to be something else speaking the standard protocol.
FROM node:22-alpine

WORKDIR /app
# The lockfile, and `npm ci` rather than `npm install`, because the ranges in
# package.json resolve differently every month: a rebuild picked up a libp2p
# newer than the multiaddr beside it and the relay died on start with
# "ma.stringTuples is not a function". Pinning is the difference between an
# image that builds and an image that builds and runs.
COPY relay/package.json relay/package-lock.json ./
RUN npm ci --omit=dev --no-audit --no-fund

COPY relay/relay.mjs ./

# The identity lives in a volume: its peer id is inside every multiaddr a
# worker holds, so regenerating it on restart invalidates them all at once.
ENV LOOM_RELAY_KEY=/data/relay/identity.key \
    LOOM_RELAY_PORT=47200

# TCP carries the reservations; UDP is QUIC, which is the transport a hole
# punch should prefer — no handshake for a firewall to answer with an RST.
EXPOSE 47200/tcp 47200/udp
CMD ["node", "relay.mjs"]
