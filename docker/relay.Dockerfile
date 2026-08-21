# libp2p circuit-relay v2 server for Loom (see docs/P2P_RELAY.md).
#
# Its own image and its own process because the workers' p2p stack, Lattica,
# is a relay CLIENT only: it announces /libp2p/circuit/relay/0.2.0/stop and
# never the /hop half a server offers. No configuration changes that, so the
# server has to be something else speaking the standard protocol.
FROM node:22-alpine

WORKDIR /app
COPY relay/package.json ./
RUN npm install --omit=dev --no-audit --no-fund

COPY relay/relay.mjs ./

# The identity lives in a volume: its peer id is inside every multiaddr a
# worker holds, so regenerating it on restart invalidates them all at once.
ENV LOOM_RELAY_KEY=/data/relay/identity.key \
    LOOM_RELAY_PORT=47200

EXPOSE 47200
CMD ["node", "relay.mjs"]
