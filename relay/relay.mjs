// A libp2p circuit-relay v2 server: the springboard Loom workers need to
// reach each other when neither of them can accept an inbound connection.
//
// Why this exists as its own process rather than a flag on the orchestrator:
// Lattica, the p2p stack the workers use, is a relay CLIENT only. It announces
//
//     /libp2p/circuit/relay/0.2.0/stop
//
// which is the protocol for RECEIVING a relayed connection, and never `.../hop`,
// which is what a relay SERVER offers. There is no configuration that changes
// that — the Rust core silently accepts unknown config keys and none of them
// turns the service on. So the relay has to be something else, speaking the
// standard protocol.
//
// What it is NOT: the path your activations travel. The orchestrator already
// relays those, permanently, and that keeps working. This relay carries a
// connection for a few seconds so two NATed peers can agree on a moment and
// punch through to each other — after which it is out of the picture. A
// destination versus a springboard.
//
// It deliberately does nothing else: no DHT, no application protocols, no
// access to model data. A relayed stream is encrypted end to end between the
// two workers (Noise), so this process cannot read what passes through it.
import { createLibp2p } from 'libp2p'
import { circuitRelayServer } from '@libp2p/circuit-relay-v2'
import { tcp } from '@libp2p/tcp'
import { identify } from '@libp2p/identify'
import { noise } from '@chainsafe/libp2p-noise'
import { yamux } from '@chainsafe/libp2p-yamux'
import { generateKeyPair, privateKeyFromProtobuf, privateKeyToProtobuf } from '@libp2p/crypto/keys'
import { readFile, writeFile, mkdir } from 'node:fs/promises'
import { dirname } from 'node:path'

const PORT = Number(process.env.LOOM_RELAY_PORT || 47200)
// The address workers are told to dial. Announced rather than detected: what
// matters is how the outside reaches us, which this host cannot know.
const PUBLIC_HOST = process.env.LOOM_RELAY_PUBLIC_HOST || ''
// The identity must survive restarts: it is inside every multiaddr a worker
// holds, so a new one on every boot invalidates them all at once.
const KEY_PATH = process.env.LOOM_RELAY_KEY || '/data/relay/identity.key'

async function loadOrCreateKey () {
  try {
    return privateKeyFromProtobuf(await readFile(KEY_PATH))
  } catch {
    const key = await generateKeyPair('Ed25519')
    await mkdir(dirname(KEY_PATH), { recursive: true })
    await writeFile(KEY_PATH, privateKeyToProtobuf(key))
    return key
  }
}

const privateKey = await loadOrCreateKey()
const node = await createLibp2p({
  privateKey,
  addresses: {
    listen: [`/ip4/0.0.0.0/tcp/${PORT}`],
    ...(PUBLIC_HOST ? { announce: [`/ip4/${PUBLIC_HOST}/tcp/${PORT}`] } : {})
  },
  transports: [tcp()],
  connectionEncrypters: [noise()],
  streamMuxers: [yamux()],
  services: {
    identify: identify(),
    relay: circuitRelayServer({
      // Generous: every worker that cannot be dialled into needs one slot for
      // as long as it is in a pipeline, and a refused reservation silently
      // costs that worker its direct path.
      reservations: {
        maxReservations: Number(process.env.LOOM_RELAY_MAX_RESERVATIONS || 512),
        reservationTtl: 60 * 60 * 1000
      }
    })
  }
})

console.log(`relay up: ${node.peerId.toString()} on port ${PORT}`)
for (const addr of node.getMultiaddrs()) console.log(`  ${addr.toString()}`)
if (!PUBLIC_HOST) {
  console.warn('LOOM_RELAY_PUBLIC_HOST is not set: workers will be handed ' +
               'addresses only reachable from this host')
}

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => { node.stop().finally(() => process.exit(0)) })
}
