"""
NEXUS Asset Universe
=====================
The complete registry of legitimate, institutionally-relevant assets
that NEXUS is designed to trade and analyze.

PHILOSOPHY:
  This system is built for serious assets with real fundamentals, real
  developer activity, real on-chain traction, and identifiable use cases.
  Scoring weights differ by category because the signals that predict
  a DeFi protocol's price action are fundamentally different from those
  that move a Layer 1 blockchain or a Real World Asset token.

CATEGORIES:
  L1_FOUNDATION       — Bitcoin: macro, store-of-value, on-chain flows dominate
  L1_SMART_CONTRACT   — ETH, SOL, AVAX, etc.: ecosystem + dev + adoption
  L2_SCALING          — ARB, OP, MATIC, etc.: fee revenue + dev + sequencer TVL
  DEFI_BLUECHIP       — AAVE, UNI, MKR, etc.: protocol revenue + TVL + tokenomics
  INFRASTRUCTURE      — LINK, GRT, FIL, etc.: integrations + dev + adoption
  CROSSCHAIN          — DOT, ATOM, RUNE: interop adoption + IBC/bridge volume
  RWA                 — ONDO, POLYX, etc.: yield, AUM, regulatory, compliance
  AI_COMPUTE          — FET, RENDER, TAO, etc.: dev velocity + compute metrics
  PAYMENTS            — XRP, XLM, HBAR, ALGO: corridor adoption + on-chain flows
  EXCHANGE_TOKEN      — BNB: exchange metrics + burn rate + ecosystem
  COMMODITY           — PAXG, XAUT: reserves, peg stability, gold price correlation
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────
# Asset Categories
# ─────────────────────────────────────────────

class AssetCategory(Enum):
    L1_FOUNDATION     = "L1 Foundation"
    L1_SMART_CONTRACT = "L1 Smart Contract"
    L2_SCALING        = "L2 Scaling"
    DEFI_BLUECHIP     = "DeFi Blue Chip"
    INFRASTRUCTURE    = "Infrastructure"
    CROSSCHAIN        = "Cross-Chain"
    RWA               = "Real World Asset"
    AI_COMPUTE        = "AI / Compute"
    PAYMENTS          = "Payments"
    EXCHANGE_TOKEN    = "Exchange Token"
    COMMODITY         = "Commodity-Backed"


# ─────────────────────────────────────────────
# Per-Category Scoring Weight Profiles
# ─────────────────────────────────────────────

WEIGHT_PROFILES: dict[AssetCategory, dict[str, float]] = {

    AssetCategory.L1_FOUNDATION: {
        # BTC: macro + on-chain flows are 65% of the signal.
        # Dev activity is near-irrelevant — protocol is ossified by design.
        # MVRV, SOPR, exchange reserve flows, whale accumulation are king.
        "technical":  0.25,
        "onchain":    0.40,
        "dev":        0.05,
        "sentiment":  0.15,
        "tokenomics": 0.15,
    },

    AssetCategory.L1_SMART_CONTRACT: {
        # ETH/SOL/AVAX: ecosystem adoption + developer velocity matter most.
        # On-chain: gas fees, active addresses, DApp revenue tell the real story.
        # Dev: GitHub commit velocity predicts major protocol upgrades 4-6 weeks early.
        "technical":  0.20,
        "onchain":    0.25,
        "dev":        0.25,
        "sentiment":  0.15,
        "tokenomics": 0.15,
    },

    AssetCategory.L2_SCALING: {
        # ARB/OP/MATIC: fee revenue per transaction, sequencer TVL, and bridge inflows.
        # Dev velocity is key — L2s differentiate on tech (OP Stack, zkEVM, Nitro).
        # On-chain: sequencer revenue, number of active rollups, bridge TVL.
        "technical":  0.20,
        "onchain":    0.25,
        "dev":        0.25,
        "sentiment":  0.15,
        "tokenomics": 0.15,
    },

    AssetCategory.DEFI_BLUECHIP: {
        # AAVE/UNI/MKR/CRV/LDO: protocol revenue and TVL are the fundamental anchors.
        # Tokenomics: fee switch status, buyback/burn mechanisms, governance voting.
        # On-chain: borrow rates, utilization ratios, collateral health.
        # Sentiment weighted lower — DeFi moves on fundamentals more than narrative.
        "technical":  0.20,
        "onchain":    0.30,
        "dev":        0.15,
        "sentiment":  0.10,
        "tokenomics": 0.25,
    },

    AssetCategory.INFRASTRUCTURE: {
        # LINK/GRT/FIL/ICP: adoption by other protocols is the primary value driver.
        # Dev: integration count and commit velocity signal ecosystem health.
        # On-chain: number of data feeds live (LINK), indexed subgraphs (GRT).
        # Tokenomics: staking participation rate, node operator economics.
        "technical":  0.20,
        "onchain":    0.20,
        "dev":        0.30,
        "sentiment":  0.15,
        "tokenomics": 0.15,
    },

    AssetCategory.CROSSCHAIN: {
        # DOT/ATOM/RUNE/AXL: IBC/bridge transaction volume is primary signal.
        # Dev: parachain/appchain launches (DOT), zone growth (ATOM).
        # Tokenomics: security budget (staking ratio), inflation vs. adoption.
        "technical":  0.20,
        "onchain":    0.20,
        "dev":        0.25,
        "sentiment":  0.15,
        "tokenomics": 0.20,
    },

    AssetCategory.RWA: {
        # ONDO/POLYX/CFG: yield and AUM are the primary attractors of capital.
        # Regulatory clarity is a massive binary catalyst — monitor closely.
        # Tokenomics: APY vs. competing yields (T-bills, money markets).
        # Sentiment weighted very low — institutional buyers aren't Twitter-driven.
        "technical":  0.20,
        "onchain":    0.20,
        "dev":        0.15,
        "sentiment":  0.10,
        "tokenomics": 0.35,
    },

    AssetCategory.AI_COMPUTE: {
        # FET/RENDER/TAO/OCEAN: narrative + developer velocity are co-equal drivers.
        # Dev: GitHub activity and model releases are actual product milestones.
        # Sentiment: AI narrative cycles with macro tech sentiment (NVDA, AI headlines).
        # Tokenomics: compute utilization rate, revenue from actual inference/rendering.
        "technical":  0.20,
        "onchain":    0.15,
        "dev":        0.30,
        "sentiment":  0.20,
        "tokenomics": 0.15,
    },

    AssetCategory.PAYMENTS: {
        # XRP/XLM/HBAR/ALGO: corridor adoption and regulatory status are key.
        # On-chain: ODL volume (XRP), payment corridor activity, settlement speed.
        # Dev: CBDC partnerships, bank integrations, enterprise pipeline.
        # Tokenomics: escrow unlock schedule (XRP), institutional treasury context.
        "technical":  0.20,
        "onchain":    0.30,
        "dev":        0.15,
        "sentiment":  0.15,
        "tokenomics": 0.20,
    },

    AssetCategory.EXCHANGE_TOKEN: {
        # BNB: exchange trading volume → burn rate → supply reduction.
        # On-chain: BNB burn events, BNBChain TVL, BEP-20 activity.
        # Tokenomics: quarterly burn schedule and cumulative burn statistics.
        "technical":  0.25,
        "onchain":    0.25,
        "dev":        0.15,
        "sentiment":  0.15,
        "tokenomics": 0.20,
    },

    AssetCategory.COMMODITY: {
        # PAXG/XAUT: peg to underlying commodity price is the primary signal.
        # Tokenomics: reserve audit status, redemption mechanism health.
        # Technical: follows spot gold/silver — use TradFi correlation.
        # Dev essentially irrelevant — these are tokenization wrappers.
        "technical":  0.30,
        "onchain":    0.20,
        "dev":        0.05,
        "sentiment":  0.10,
        "tokenomics": 0.35,
    },
}


# ─────────────────────────────────────────────
# Asset Registry
# ─────────────────────────────────────────────

@dataclass
class Asset:
    symbol:        str              # CCXT trading pair, e.g. "BTC/USDT"
    name:          str              # Human name
    category:      AssetCategory
    coingecko_id:  str              # For CoinGecko API
    defi_protocol: str              # DeFiLlama slug (empty if N/A)
    github_repo:   str              # "owner/repo" for dev activity (empty if N/A)
    tokterm_id:    str              # Token Terminal project ID (empty if N/A)
    description:   str              # One-line thesis
    tags:          list[str] = field(default_factory=list)

    @property
    def weights(self) -> dict[str, float]:
        return WEIGHT_PROFILES[self.category]

    @property
    def base(self) -> str:
        return self.symbol.split("/")[0]


# ─────────────────────────────────────────────
# The Full Asset Registry
# ─────────────────────────────────────────────

REGISTRY: list[Asset] = [

    # ── L1 Foundation ─────────────────────────────────────────────
    Asset("BTC/USDT",  "Bitcoin",    AssetCategory.L1_FOUNDATION,
          "bitcoin",   "",           "bitcoin/bitcoin", "bitcoin",
          "Digital store of value. Macro crypto benchmark. Institutional on-ramp.",
          ["store-of-value", "macro", "institutional", "pow", "hard-cap"]),

    # ── L1 Smart Contract ─────────────────────────────────────────
    Asset("ETH/USDT",  "Ethereum",   AssetCategory.L1_SMART_CONTRACT,
          "ethereum",  "ethereum",   "ethereum/go-ethereum", "ethereum",
          "Dominant smart contract platform. DeFi and NFT base layer. EIP-1559 burn.",
          ["smart-contract", "defi-base", "deflationary", "pos", "evm"]),

    Asset("SOL/USDT",  "Solana",     AssetCategory.L1_SMART_CONTRACT,
          "solana",    "solana",     "solana-labs/solana", "solana",
          "High-performance L1. Fast finality, low fees. Strong DeFi + consumer app ecosystem.",
          ["smart-contract", "high-throughput", "defi", "nft", "consumer-apps"]),

    Asset("BNB/USDT",  "BNB",        AssetCategory.EXCHANGE_TOKEN,
          "binancecoin","bnb-chain", "bnb-chain/bsc", "bnb",
          "Binance ecosystem token. Quarterly burn from exchange revenue. BNBChain TVL.",
          ["exchange-token", "burn-mechanism", "bnbchain", "evm"]),

    Asset("AVAX/USDT", "Avalanche",  AssetCategory.L1_SMART_CONTRACT,
          "avalanche-2","avalanche", "ava-labs/avalanchego", "avalanche",
          "Subnet architecture for institutional chains. Fast finality. Subnet launches.",
          ["smart-contract", "subnets", "institutional", "evm", "high-speed"]),

    Asset("NEAR/USDT", "NEAR Protocol", AssetCategory.L1_SMART_CONTRACT,
          "near",      "near",       "near/nearcore", "near-protocol",
          "Sharding-native L1. Chain abstraction narrative. NEAR AI integration.",
          ["smart-contract", "sharding", "chain-abstraction", "ai-integration"]),

    Asset("APT/USDT",  "Aptos",      AssetCategory.L1_SMART_CONTRACT,
          "aptos",     "aptos",      "aptos-labs/aptos-core", "aptos",
          "Move language L1. Backed by major VCs. Strong developer ecosystem growth.",
          ["smart-contract", "move-language", "vc-backed", "parallel-execution"]),

    Asset("SUI/USDT",  "Sui",        AssetCategory.L1_SMART_CONTRACT,
          "sui",       "sui",        "MystenLabs/sui", "sui",
          "Move language L1. Object-centric model. Mysten Labs infrastructure.",
          ["smart-contract", "move-language", "object-model", "parallel-execution"]),

    Asset("TON/USDT",  "Toncoin",    AssetCategory.L1_SMART_CONTRACT,
          "the-open-network","ton",  "ton-blockchain/ton", "",
          "Telegram-native blockchain. 900M+ potential user distribution advantage.",
          ["smart-contract", "telegram", "mass-adoption", "messaging-wallet"]),

    Asset("HBAR/USDT", "Hedera",     AssetCategory.PAYMENTS,
          "hedera-hashgraph","hedera","hashgraph/hedera-services","",
          "Enterprise-grade hashgraph. Governed council includes Google, IBM, Boeing.",
          ["enterprise", "hashgraph", "council-governed", "cbdc-adjacent", "esg"]),

    Asset("ALGO/USDT", "Algorand",   AssetCategory.PAYMENTS,
          "algorand",  "",           "algorand/go-algorand","",
          "Pure PoS. Carbon-negative. Institutional and government blockchain programs.",
          ["pure-pos", "carbon-negative", "institutional", "government-partnerships"]),

    # ── L2 Scaling ────────────────────────────────────────────────
    Asset("ARB/USDT",  "Arbitrum",   AssetCategory.L2_SCALING,
          "arbitrum",  "arbitrum",   "OffchainLabs/nitro", "arbitrum",
          "Largest L2 by TVL. Arbitrum One + Nova + Orbit. Deep DeFi ecosystem.",
          ["l2", "optimistic-rollup", "highest-tvl", "defi-hub", "nitro"]),

    Asset("OP/USDT",   "Optimism",   AssetCategory.L2_SCALING,
          "optimism",  "optimism",   "ethereum-optimism/optimism", "optimism",
          "OP Stack foundation. Superchain architecture. Coinbase Base runs on OP Stack.",
          ["l2", "optimistic-rollup", "op-stack", "superchain", "base"]),

    Asset("MATIC/USDT","Polygon",    AssetCategory.L2_SCALING,
          "matic-network","polygon", "0xPolygon/polygon-edge","polygon",
          "Multi-scaling framework. zkEVM + CDK. Strong enterprise/TradFi partnerships.",
          ["l2", "zkevm", "cdk", "enterprise", "tradfi-partnerships"]),

    Asset("IMX/USDT",  "ImmutableX", AssetCategory.L2_SCALING,
          "immutable-x","immutablex","immutable/imx-core-sdk","",
          "L2 purpose-built for gaming and NFTs. Zero gas fees. Immutable zkEVM.",
          ["l2", "gaming", "nft", "zero-gas", "zkevm"]),

    Asset("STRK/USDT", "Starknet",   AssetCategory.L2_SCALING,
          "starknet",  "starknet",   "starkware-libs/cairo", "",
          "ZK rollup with Cairo VM. Native account abstraction. Ethereum security.",
          ["l2", "zk-rollup", "cairo", "account-abstraction", "ethereum-secured"]),

    # ── DeFi Blue Chips ───────────────────────────────────────────
    Asset("AAVE/USDT", "Aave",       AssetCategory.DEFI_BLUECHIP,
          "aave",      "aave",       "aave/aave-v3-core","aave",
          "Largest decentralized lending protocol. Real revenue, multi-chain, GHO stablecoin.",
          ["defi", "lending", "real-revenue", "multi-chain", "stablecoin"]),

    Asset("UNI/USDT",  "Uniswap",    AssetCategory.DEFI_BLUECHIP,
          "uniswap",   "uniswap",    "Uniswap/v4-core","uniswap",
          "Dominant DEX. $2T+ cumulative volume. UniswapX, V4 hooks, fee switch debate.",
          ["defi", "dex", "amm", "dominant", "fee-switch-potential"]),

    Asset("MKR/USDT",  "MakerDAO",   AssetCategory.DEFI_BLUECHIP,
          "maker",     "makerdao",   "makerdao/dss","maker",
          "DAI issuer. Pioneering RWA integration. DSR yield. Endgame restructure.",
          ["defi", "stablecoin", "rwa-leader", "real-revenue", "endgame"]),

    Asset("LDO/USDT",  "Lido",       AssetCategory.DEFI_BLUECHIP,
          "lido-dao",  "lido",       "lidofinance/core","lido",
          "Largest liquid staking protocol. 32%+ of staked ETH. stETH composability.",
          ["defi", "liquid-staking", "dominant", "eth-staking", "composable"]),

    Asset("RPL/USDT",  "Rocket Pool",AssetCategory.DEFI_BLUECHIP,
          "rocket-pool","rocket-pool","rocket-pool/rocketpool","",
          "Decentralized ETH staking. rETH. Lower counterparty risk vs Lido.",
          ["defi", "liquid-staking", "decentralized", "eth-staking", "reth"]),

    Asset("GMX/USDT",  "GMX",        AssetCategory.DEFI_BLUECHIP,
          "gmx",       "gmx",        "gmx-io/gmx-contracts","gmx",
          "Leading perp DEX. Real yield to stakers. GLP liquidity pool model.",
          ["defi", "perp-dex", "real-yield", "arbitrum", "glp"]),

    Asset("PENDLE/USDT","Pendle",    AssetCategory.DEFI_BLUECHIP,
          "pendle",    "pendle",     "pendalfinance/pendle-core-v2","",
          "Yield tokenization. Separates principal from yield. Critical for RWA layer.",
          ["defi", "yield-tokenization", "rwa-adjacent", "fixed-income", "innovative"]),

    Asset("CRV/USDT",  "Curve",      AssetCategory.DEFI_BLUECHIP,
          "curve-dao-token","curve", "curvefi/curve-contract","curve",
          "Stablecoin DEX. veToken pioneer. $3B+ TVL. crvUSD stablecoin.",
          ["defi", "stablecoin-dex", "vetoken", "real-revenue", "crvusd"]),

    # ── Infrastructure / Oracle ───────────────────────────────────
    Asset("LINK/USDT", "Chainlink",  AssetCategory.INFRASTRUCTURE,
          "chainlink", "",           "smartcontractkit/chainlink","",
          "Oracle infrastructure for all of DeFi. CCIP cross-chain messaging. DECO privacy.",
          ["oracle", "infrastructure", "ccip", "defi-critical", "staking-v2"]),

    Asset("GRT/USDT",  "The Graph",  AssetCategory.INFRASTRUCTURE,
          "the-graph", "",           "graphprotocol/graph-node","",
          "Indexing protocol. The Google of blockchain data. GraphQL queries for all chains.",
          ["infrastructure", "indexing", "data", "decentralized", "graphql"]),

    Asset("FIL/USDT",  "Filecoin",   AssetCategory.INFRASTRUCTURE,
          "filecoin",  "",           "filecoin-project/lotus","",
          "Decentralized storage network. 20+ EiB of capacity. Real utility use cases.",
          ["infrastructure", "storage", "decentralized", "web3", "real-capacity"]),

    Asset("ICP/USDT",  "Internet Computer", AssetCategory.INFRASTRUCTURE,
          "internet-computer","",   "dfinity/ic","",
          "On-chain computation at internet scale. Chain key cryptography. Web3 hosting.",
          ["infrastructure", "computation", "web3-hosting", "dfinity", "chain-key"]),

    Asset("WLD/USDT",  "Worldcoin",  AssetCategory.INFRASTRUCTURE,
          "worldcoin-wld","",       "worldcoin/world-id-contracts","",
          "Proof-of-personhood via iris biometrics. Sam Altman project. Identity layer.",
          ["infrastructure", "identity", "proof-of-personhood", "sam-altman", "biometrics"]),

    # ── Cross-Chain / Interoperability ────────────────────────────
    Asset("DOT/USDT",  "Polkadot",   AssetCategory.CROSSCHAIN,
          "polkadot",  "",           "paritytech/polkadot","polkadot",
          "Relay chain + parachain architecture. Shared security. 100+ parachains live.",
          ["crosschain", "parachain", "shared-security", "substrate", "governance"]),

    Asset("ATOM/USDT", "Cosmos",     AssetCategory.CROSSCHAIN,
          "cosmos",    "",           "cosmos/gaia","cosmos",
          "IBC protocol standard. App-chain thesis. 100+ zones. $10B+ IBC volume.",
          ["crosschain", "ibc", "app-chain", "interoperability", "pos"]),

    Asset("RUNE/USDT", "THORChain",  AssetCategory.CROSSCHAIN,
          "thorchain", "thorchain",  "thorchain/thornode","",
          "Native cross-chain swaps without wrapping. Real revenue, real liquidity.",
          ["crosschain", "native-swaps", "real-revenue", "decentralized", "no-wrapping"]),

    Asset("AXL/USDT",  "Axelar",     AssetCategory.CROSSCHAIN,
          "axelar",    "axelar",     "axelarnetwork/axelar-core","",
          "Generalized cross-chain messaging. Used by USDC, Microsoft, Uniswap.",
          ["crosschain", "messaging", "usdc-bridge", "institutional-adoption"]),

    Asset("TIA/USDT",  "Celestia",   AssetCategory.CROSSCHAIN,
          "celestia",  "",           "celestiaorg/celestia-node","",
          "Modular data availability layer. Enables cheap rollup deployment.",
          ["crosschain", "data-availability", "modular", "rollup-infrastructure"]),

    # ── Real World Assets ─────────────────────────────────────────
    Asset("ONDO/USDT", "Ondo Finance", AssetCategory.RWA,
          "ondo-finance","ondo",    "ondofinance/ondo-v1","",
          "Tokenized US Treasuries (USDY, OUSG). $600M+ AUM. BlackRock partnership.",
          ["rwa", "tokenized-treasuries", "yield", "institutional", "blackrock"]),

    Asset("POLYX/USDT","Polymesh",   AssetCategory.RWA,
          "polymesh",  "",           "PolymathNetwork/polymesh-sdk","",
          "Purpose-built regulated securities blockchain. SEC/FCA engagement.",
          ["rwa", "regulated-securities", "compliance", "institutional", "kyc-native"]),

    Asset("CFG/USDT",  "Centrifuge", AssetCategory.RWA,
          "centrifuge","centrifuge","centrifuge/centrifuge-chain","",
          "On-chain real-world lending. Tokenized invoices and trade finance.",
          ["rwa", "real-world-lending", "defi", "trade-finance", "tokenization"]),

    # ── AI / Compute ──────────────────────────────────────────────
    Asset("FET/USDT",  "Fetch.ai",   AssetCategory.AI_COMPUTE,
          "fetch-ai",  "",           "fetchai/fetchd","",
          "Autonomous AI agents on-chain. ASI Alliance merger with AGIX and OCEAN.",
          ["ai", "agents", "compute", "asi-alliance", "multi-agent"]),

    Asset("RENDER/USDT","Render",    AssetCategory.AI_COMPUTE,
          "render-token","",        "rendernetwork/contracts","",
          "Decentralized GPU rendering. AI workloads. Partnership with NVIDIA.",
          ["ai", "compute", "gpu", "nvidia-partnership", "rendering"]),

    Asset("TAO/USDT",  "Bittensor",  AssetCategory.AI_COMPUTE,
          "bittensor", "",           "opentensor/subtensor","",
          "Decentralized AI network. Subnet marketplace. Incentivized ML training.",
          ["ai", "machine-learning", "decentralized", "subnets", "incentivized-ml"]),

    Asset("OCEAN/USDT","Ocean Protocol", AssetCategory.AI_COMPUTE,
          "ocean-protocol","",      "oceanprotocol/ocean.py","",
          "Data marketplace and monetization. Compute-to-data privacy. AI training data.",
          ["ai", "data-marketplace", "privacy", "compute-to-data", "asi-alliance"]),

    # ── Payments / Cross-Border ───────────────────────────────────
    Asset("XRP/USDT",  "XRP",        AssetCategory.PAYMENTS,
          "ripple",    "",           "XRPLF/rippled","",
          "ODL cross-border payments. SEC case resolved. CBDC infrastructure candidate. XRPL AMM.",
          ["payments", "cross-border", "cbdc", "institutional", "odl", "xrpl-amm"]),

    Asset("XLM/USDT",  "Stellar",    AssetCategory.PAYMENTS,
          "stellar",   "",           "stellar/stellar-core","",
          "Non-profit-governed payment network. MoneyGram partnership. Anchored assets.",
          ["payments", "cross-border", "nonprofit", "moneygram", "anchored-assets"]),

    # ── Commodity-Backed ──────────────────────────────────────────
    Asset("PAXG/USDT", "PAX Gold",   AssetCategory.COMMODITY,
          "pax-gold",  "",           "","",
          "1 PAXG = 1 fine troy ounce of London Good Delivery gold. Audited reserves.",
          ["commodity", "gold", "tokenized", "audited", "paxos"]),
]


# ─────────────────────────────────────────────
# Lookup Helpers
# ─────────────────────────────────────────────

_by_symbol: dict[str, Asset] = {a.symbol: a for a in REGISTRY}
_by_base:   dict[str, Asset] = {a.base: a   for a in REGISTRY}

def get(symbol: str) -> Optional[Asset]:
    """Look up an Asset by trading pair (e.g. 'BTC/USDT') or base (e.g. 'BTC')."""
    return _by_symbol.get(symbol) or _by_base.get(symbol)

def by_category(cat: AssetCategory) -> list[Asset]:
    return [a for a in REGISTRY if a.category == cat]

def symbols_for_category(cat: AssetCategory) -> list[str]:
    return [a.symbol for a in by_category(cat)]

def all_symbols() -> list[str]:
    return [a.symbol for a in REGISTRY]

def weights_for(symbol: str) -> dict[str, float]:
    """Return category-specific scoring weights, or balanced defaults."""
    asset = get(symbol)
    if asset:
        return asset.weights
    return {"technical":0.25,"onchain":0.25,"dev":0.20,"sentiment":0.15,"tokenomics":0.15}


# ─────────────────────────────────────────────
# Curated Watchlists for Different Use Cases
# ─────────────────────────────────────────────

class Watchlist:
    """
    Pre-built watchlists for common trading strategies.
    All contain only liquid, institutionally-relevant assets.
    """

    # ── Breadth scan: broad market health (30 liquid assets) ──────
    BREADTH_UNIVERSE: list[str] = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
        "AVAX/USDT","ARB/USDT", "OP/USDT",  "MATIC/USDT","DOT/USDT",
        "LINK/USDT","ATOM/USDT","NEAR/USDT","APT/USDT",  "SUI/USDT",
        "AAVE/USDT","UNI/USDT", "MKR/USDT", "LDO/USDT",  "CRV/USDT",
        "INJ/USDT", "GMX/USDT", "TIA/USDT", "RUNE/USDT", "ICP/USDT",
        "FET/USDT", "RENDER/USDT","TAO/USDT","ONDO/USDT","PENDLE/USDT",
    ]

    # ── Core portfolio: highest-conviction liquid assets ──────────
    CORE_PORTFOLIO: list[str] = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT",
        "LINK/USDT","AVAX/USDT","ARB/USDT", "AAVE/USDT","DOT/USDT",
    ]

    # ── DeFi focus: protocol revenue and TVL-driven trades ────────
    DEFI_FOCUS: list[str] = [
        "AAVE/USDT","UNI/USDT", "MKR/USDT", "LDO/USDT", "RPL/USDT",
        "GMX/USDT", "CRV/USDT", "PENDLE/USDT","RUNE/USDT","BAL/USDT",
    ]

    # ── XRPL ecosystem ────────────────────────────────────────────
    XRPL_ECOSYSTEM: list[str] = [
        "XRP/USDT",
        # XRPL-native tokens fetched via XRPLBreadthCalculator + FirstLedger
        # Add your XRPL DEX tokens here as discovered from dexscreener.com/xrpl
    ]

    # ── Infrastructure focus ──────────────────────────────────────
    INFRASTRUCTURE: list[str] = [
        "LINK/USDT","GRT/USDT", "FIL/USDT", "ICP/USDT",
        "AXL/USDT", "TIA/USDT", "WLD/USDT", "RUNE/USDT",
    ]

    # ── RWA + institutional ───────────────────────────────────────
    RWA_INSTITUTIONAL: list[str] = [
        "ONDO/USDT","POLYX/USDT","CFG/USDT", "MKR/USDT",
        "PENDLE/USDT","HBAR/USDT","ALGO/USDT","PAXG/USDT",
    ]

    # ── L2 scaling race ───────────────────────────────────────────
    L2_RACE: list[str] = [
        "ARB/USDT","OP/USDT","MATIC/USDT","IMX/USDT","STRK/USDT",
    ]

    # ── AI / compute narrative ────────────────────────────────────
    AI_COMPUTE: list[str] = [
        "FET/USDT","RENDER/USDT","TAO/USDT","OCEAN/USDT","WLD/USDT",
    ]

    # ── Payments corridor ─────────────────────────────────────────
    PAYMENTS: list[str] = [
        "XRP/USDT","XLM/USDT","HBAR/USDT","ALGO/USDT",
    ]


# ─────────────────────────────────────────────
# Category Summary
# ─────────────────────────────────────────────

def print_universe_summary():
    from collections import Counter
    counts = Counter(a.category.value for a in REGISTRY)
    print("\n  NEXUS ASSET UNIVERSE\n  " + "─" * 50)
    for cat, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:<26} {count:>2} assets")
    print(f"  {'─' * 38}")
    print(f"  {'Total':<26} {len(REGISTRY):>2} assets\n")

    print("  Category scoring profiles (TA / OC / DEV / SENT / TOK):")
    for cat in AssetCategory:
        w = WEIGHT_PROFILES[cat]
        print(f"  {cat.value:<26} "
              f"{w['technical']*100:.0f}% / "
              f"{w['onchain']*100:.0f}% / "
              f"{w['dev']*100:.0f}% / "
              f"{w['sentiment']*100:.0f}% / "
              f"{w['tokenomics']*100:.0f}%")


if __name__ == "__main__":
    print_universe_summary()
    print("\nBreadth universe symbols:")
    print(Watchlist.BREADTH_UNIVERSE)
