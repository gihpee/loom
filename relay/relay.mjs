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
import { quic } from '@chainsafe/libp2p-quic'
import { identify } from '@libp2p/identify'
import { ping } from '@libp2p/ping'
import { noise } from '@chainsafe/libp2p-noise'
import { yamux } from '@chainsafe/libp2p-yamux'
import { generateKeyPair, privateKeyFromProtobuf, privateKeyToProtobuf } from '@libp2p/crypto/keys'
import { readFile, writeFile, mkdir } from 'node:fs/promises'
import { dirname, join } from 'node:path'

const PORT = Number(process.env.LOOM_RELAY_PORT || 47200)

// The address workers are told to dial. Announced rather than detected: what
// matters is how the outside reaches us, which this host cannot know.
//
// A host, not a multiaddr — but the multiaddr is what this relay prints, so
// pasting THAT back in is the natural mistake, and it used to end in a crash
// deep inside the address parser ("Protocol 201.34.135.177 was unknown"),
// naming the value and not the setting. Both forms are accepted now.
function announceHost (raw) {
  const value = (raw || '').trim()
  if (!value.startsWith('/')) return value
  const parts = value.split('/')
  const at = parts.findIndex(p => ['ip4', 'ip6', 'dns', 'dns4', 'dns6'].includes(p))
  return at >= 0 ? (parts[at + 1] || '') : ''
}

const RAW_HOST = process.env.LOOM_RELAY_PUBLIC_HOST || ''
const PUBLIC_HOST = announceHost(RAW_HOST)
if (RAW_HOST.startsWith('/') && PUBLIC_HOST) {
  console.warn(`LOOM_RELAY_PUBLIC_HOST is a multiaddr; using its host: ${PUBLIC_HOST}`)
}
if (RAW_HOST && !PUBLIC_HOST) {
  console.error(`LOOM_RELAY_PUBLIC_HOST=${RAW_HOST} has no host in it. ` +
                'Set it to the address workers reach this machine at, e.g. 203.0.113.7')
  process.exit(2)
}
// A name needs /dns4, an address needs /ip4 — announcing a hostname under
// /ip4 fails the same way, one layer down.
const HOST_PROTO = /^\d+\.\d+\.\d+\.\d+$/.test(PUBLIC_HOST)
  ? 'ip4'
  : (PUBLIC_HOST.includes(':') ? 'ip6' : 'dns4')
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
    // Both transports on the same number. QUIC matters more than it looks:
    // hole punching over TCP needs a simultaneous open, which many stateful
    // firewalls answer with an RST, while QUIC is plain UDP with no handshake
    // to interrupt. A relay that only speaks TCP forces every negotiation it
    // hosts down the harder path.
    listen: [`/ip4/0.0.0.0/tcp/${PORT}`, `/ip4/0.0.0.0/udp/${PORT}/quic-v1`],
    ...(PUBLIC_HOST
      ? {
          announce: [
            `/${HOST_PROTO}/${PUBLIC_HOST}/tcp/${PORT}`,
            `/${HOST_PROTO}/${PUBLIC_HOST}/udp/${PORT}/quic-v1`
          ]
        }
      : {})
  },
  transports: [tcp(), quic()],
  connectionEncrypters: [noise()],
  streamMuxers: [yamux()],
  services: {
    identify: identify(),
    // Not decoration. Lattica pings whatever it is connected to, and a peer
    // that answers "Unsupported" is dropped as dead — taking the reservation
    // with it. Without this the relay works for about ten seconds, then the
    // client reconnects, pings, is disappointed, and drops again, forever:
    //
    //   WARN lattica: Ping failed for peer 12D3KooW...: Unsupported
    //   (visible addrs: circuit address present, then gone)
    //
    // A relay whose reservations keep evaporating is worse than none, because
    // everything else about it looks healthy.
    ping: ping(),
    relay: circuitRelayServer({
      // Generous: every worker that cannot be dialled into needs one slot for
      // as long as it is in a pipeline, and a refused reservation silently
      // costs that worker its direct path.
      reservations: {
        maxReservations: Number(process.env.LOOM_RELAY_MAX_RESERVATIONS || 512),
        reservationTtl: 60 * 60 * 1000,
        // The standard v2 limits, kept ON DELIBERATELY: 128 KB and two minutes
        // per relayed connection. They are what stops this from becoming the
        // data path — and it must not be one. Activations relayed here would
        // take the same two wide-area crossings as the orchestrator's tunnel
        // plus a general-purpose relay in the middle; measured on a two-stage
        // pipeline, transport per token went 200 ms -> 320 ms that way.
        //
        // For a 4B model 128 KB is about 25 tokens, so a pipeline that tried
        // it would also be torn down and re-established mid-generation. Raising
        // the limit hides that symptom and keeps the slower path; the worker
        // instead measures the link and declines to use it (p2p/links.py).
        applyDefaultLimit: true
      }
    })
  }
})

console.log(`relay up: ${node.peerId.toString()} on port ${PORT}`)
for (const addr of node.getMultiaddrs()) console.log(`  ${addr.toString()}`)

// Hand the address to the orchestrator directly. It used to be a human step —
// copy the multiaddr out of this log, into .env, restart the orchestrator —
// and skipping it leaves everything looking healthy: the relay runs, the
// workers run, and the only trace is "(2 bootstrap, 0 relay)" in a worker log.
// The two processes share a volume, so they can just agree on a file.
//
// Only written when the public host is known. An address of a container's own
// interface is worse than none: a worker elsewhere would reserve a slot it can
// never be reached through.
const ADDR_FILE = process.env.LOOM_RELAY_ADDR_FILE || join(dirname(KEY_PATH), 'address')
if (PUBLIC_HOST) {
  // TCP first, and that ordering is load-bearing: measured against this very
  // relay, Lattica requests a reservation over the TCP address immediately
  // and never over the QUIC one — it connects over QUIC happily and simply
  // sends no RESERVE. Handing it both is safe (it picks TCP), handing it QUIC
  // alone leaves the worker with no slot at all.
  const announced = node.getMultiaddrs()
    .map(a => a.toString())
    .filter(a => a.includes(`/${PUBLIC_HOST}/`))
    .sort((a, b) => (a.includes('/quic') ? 1 : 0) - (b.includes('/quic') ? 1 : 0))
  await mkdir(dirname(ADDR_FILE), { recursive: true })
  await writeFile(ADDR_FILE, announced.join('\n') + '\n')
  console.log(`address published for the orchestrator: ${ADDR_FILE}`)
} else {
  console.warn('LOOM_RELAY_PUBLIC_HOST is not set: workers will be handed ' +
               'addresses only reachable from this host, so none is published')
}

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => { node.stop().finally(() => process.exit(0)) })
}
