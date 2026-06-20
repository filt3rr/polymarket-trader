"""
Polygon wallet — read-only balance and connectivity checks.
The private key never leaves this module and is never logged.
"""
import logging

logger = logging.getLogger(__name__)

# Native USDC on Polygon (Circle's canonical USDC, used by Polymarket since 2024).
# USDC.e (bridged) address: 0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174
USDC_POLYGON = "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359"

_ERC20_ABI = [
    {
        "name": "balanceOf",
        "type": "function",
        "stateMutability": "view",
        "inputs": [{"name": "account", "type": "address"}],
        "outputs": [{"name": "", "type": "uint256"}],
    },
]


class PolygonWallet:
    """
    Wraps a Polygon EOA for balance reads and address derivation.
    No write operations here — order signing is handled by py-clob-client.
    """

    def __init__(self, private_key: str, rpc_url: str = "https://polygon-rpc.com") -> None:
        from web3 import Web3
        try:
            from web3.middleware import ExtraDataToPOAMiddleware
            _poa_middleware = ExtraDataToPOAMiddleware
        except ImportError:
            # web3 <6 used geth_poa_middleware
            from web3.middleware import geth_poa_middleware  # type: ignore
            _poa_middleware = geth_poa_middleware  # type: ignore

        self._w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 10}))
        try:
            self._w3.middleware_onion.inject(_poa_middleware, layer=0)
        except Exception:
            pass  # Some RPC providers don't need POA middleware

        self._account = self._w3.eth.account.from_key(private_key)
        self._usdc = self._w3.eth.contract(
            address=Web3.to_checksum_address(USDC_POLYGON),
            abi=_ERC20_ABI,
        )
        logger.info("Wallet loaded: %s", self.address_masked)

    @property
    def address(self) -> str:
        return self._account.address

    @property
    def address_masked(self) -> str:
        a = self._account.address
        return f"{a[:6]}...{a[-4:]}"

    def usdc_balance(self) -> float:
        """USDC balance in dollars (USDC has 6 decimals on Polygon)."""
        raw = self._usdc.functions.balanceOf(self._account.address).call()
        return raw / 1_000_000

    def pol_balance(self) -> float:
        """POL/MATIC balance for gas."""
        from web3 import Web3
        raw = self._w3.eth.get_balance(self._account.address)
        return float(Web3.from_wei(raw, "ether"))

    def is_connected(self) -> bool:
        try:
            return self._w3.is_connected()
        except Exception:
            return False
