import asyncio
import hashlib
import json
from functools import wraps
import os
from contextlib import asynccontextmanager


from typing import Dict, Any, cast
from aiocache import cached
import sys
import os
import aiofiles
from eth_typing import Address, ChecksumAddress, HexStr


from pydantic import BaseModel
from web3 import Web3, HTTPProvider
from web3.types import Nonce, TxParams, TxReceipt, Wei
from anvil_web3 import AnvilWeb3, AnvilInstance
from web3.contract.contract import Contract

from utils import ContractArtifact, ReservationState, SwapReservation

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from circuits.utils.noir_lib import normalize_hex_str, run_command
from circuits.utils.rift_lib import LiquidityProvider, build_giga_circuit_proof_and_input
from circuits.utils.proxy_wallet import wei_to_satoshi
from circuits.utils.btc_data import get_rift_btc_data

@asynccontextmanager
async def change_dir_async(new_dir):
    original_dir = os.getcwd()
    try:
        os.chdir(new_dir)
        yield
    finally:
        os.chdir(original_dir)


def testnet_autoexit(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        anvil = AnvilInstance(port="9000", supress_anvil_output=True)

        kwargs["anvil"] = anvil
        try:
            return await func(*args, **kwargs)
        finally:
            anvil.kill()

    return wrapper


async def get_basefee_wei(w3: Web3) -> int:
    latest_block = await asyncio.to_thread(w3.eth.get_block, 'latest')
    assert "baseFeePerGas" in latest_block, latest_block
    return latest_block['baseFeePerGas']

def calculate_total_fees_wei(
    totalSwapAmount: int,
    basefee: int,
    ):
    """
    uint8 public protocolFeeBP = 100; // 100 bp = 1%
    uint256 public constant PROOF_GAS_COST = 420_000; // TODO: update to real value
    uint256 public constant RELEASE_GAS_COST = 210_000; // TODO: update to real value
    uint256 public proverReward = 0.002 ether;
    uint256 public releaserReward = 0.0002 ether;
    uint256 public constant MIN_ORDER_GAS_MULTIPLIER = 2;


    uint protocolFee = (totalSwapAmount * (protocolFeeBP / 10_000));
    uint proverFee = proverReward + ((PROOF_GAS_COST * block.basefee) * MIN_ORDER_GAS_MULTIPLIER);
    uint releaserFee = releaserReward + ((RELEASE_GAS_COST * block.basefee) * MIN_ORDER_GAS_MULTIPLIER);
    """
    protocolFeeBP = 100
    PROOF_GAS_COST = 420_000
    RELEASE_GAS_COST = 210_000
    proverReward = int(0.002 * 10**18)
    releaserReward = int(0.0002 * 10**18)
    MIN_ORDER_GAS_MULTIPLIER = 2
    protocolFee = (totalSwapAmount * (protocolFeeBP / 10_000))
    proverFee = proverReward + ((PROOF_GAS_COST * basefee) * MIN_ORDER_GAS_MULTIPLIER)
    releaserFee = releaserReward + ((RELEASE_GAS_COST * basefee) * MIN_ORDER_GAS_MULTIPLIER)
    return int(protocolFee + proverFee + releaserFee)


    


async def fund_account_with_ether(private_key: str, w3: AnvilWeb3):
    address = w3.eth.account.from_key(private_key).address
    await asyncio.to_thread(w3.anvil.set_balance, address, 10**18 * 1000)


async def send_transaction(
    w3: Web3, transaction: TxParams, private_key: str
) -> TxReceipt:
    signed_txn = w3.eth.account.sign_transaction(transaction, private_key)
    tx_hash = await asyncio.to_thread(
        w3.eth.send_raw_transaction, signed_txn.rawTransaction
    )
    tx_receipt = await asyncio.to_thread(w3.eth.wait_for_transaction_receipt, tx_hash)
    assert tx_receipt["status"] == 1, tx_receipt
    return tx_receipt


async def build_and_send_transaction(
    w3: Web3,
    account: ChecksumAddress,
    contract_function,
    private_key: str,
    value: int = 0,
) -> TxReceipt:
    nonce = await asyncio.to_thread(w3.eth.get_transaction_count, account)
    transaction = contract_function.build_transaction(
        {
            "from": account,
            "gas": 25_000_000,
            "gasPrice": Wei(w3.eth.gas_price * 2),
            "nonce": Nonce(nonce),
            "value": Wei(value),
        }
    )
    return await send_transaction(w3, transaction, private_key)


async def deposit_and_approve_weth(
    amount: int, weth: Contract, spender: str, private_key: str, w3: Web3
):
    account = w3.eth.account.from_key(private_key).address
    await build_and_send_transaction(
        w3, account, weth.functions.deposit(), private_key, amount
    )
    await build_and_send_transaction(
        w3, account, weth.functions.approve(spender, amount), private_key
    )


@cached(
    ttl=None
)  # we don't need to recompile the contracts every time this is called in a given execution
async def populate_rift_contracts_cache():
    await run_command("forge compile", cwd="contracts", strict_failure=False)


async def get_compiled_rift_exchange() -> ContractArtifact:
    await populate_rift_contracts_cache()
    async with aiofiles.open(
        "contracts/out/RiftExchange.sol/RiftExchange.json", "r"
    ) as file:
        return ContractArtifact(**json.loads(await file.read()))


async def get_compiled_weth() -> ContractArtifact:
    await populate_rift_contracts_cache()
    async with aiofiles.open("contracts/out/WETH.sol/WETH.json", "r") as file:
        return ContractArtifact(**json.loads(await file.read()))


async def get_compiled_verifier() -> ContractArtifact:
    await populate_rift_contracts_cache()
    async with aiofiles.open(
        "contracts/out/RiftPlonkVerification.sol/UltraVerifier.json", "r"
    ) as file:
        return ContractArtifact(**json.loads(await file.read()))


async def deploy_contract(
    contract_artifact: ContractArtifact,
    constructor_args: Dict[str, Any],
    private_key: str,
    w3: Web3,
) -> Contract:
    contract = w3.eth.contract(
        abi=contract_artifact.abi, bytecode=contract_artifact.bytecode.object
    )
    account = w3.eth.account.from_key(private_key).address

    tx_receipt = await build_and_send_transaction(
        w3, account, contract.constructor(**constructor_args), private_key
    )

    return cast(
        Contract,
        w3.eth.contract(
            address=tx_receipt["contractAddress"], abi=contract_artifact.abi
        ),
    )


async def deploy_verifier(private_key: str, w3: Web3) -> Contract:
    return await deploy_contract(
        contract_artifact=(await get_compiled_verifier()),
        constructor_args={},  # No arguments for verifier constructor
        private_key=private_key,
        w3=w3,
    )


async def deploy_weth(private_key: str, w3: Web3) -> Contract:
    return await deploy_contract(
        contract_artifact=await get_compiled_weth(),
        constructor_args={},  # No arguments for WETH constructor
        private_key=private_key,
        w3=w3,
    )


async def deploy_rift_exchange(
    initial_checkpoint_height: int,
    initial_block_hash: str,
    verifier_contract_address: str,
    deposit_token_address: str,
    private_key: str,
    w3: Web3,
) -> Contract:
    if not initial_block_hash.startswith("0x"):
        initial_block_hash = "0x" + initial_block_hash
    return await deploy_contract(
        contract_artifact=await get_compiled_rift_exchange(),
        constructor_args={
            "initialCheckpointHeight": initial_checkpoint_height,
            "initialBlockHash": initial_block_hash,
            "verifierContractAddress": verifier_contract_address,
            "depositTokenAddress": deposit_token_address,
        },
        private_key=private_key,
        w3=w3,
    )
    

async def reserve_liquidity_rift(
    vault_indexes_to_reserve: list[int],
    amounts_to_reserve: list[int],
    total_swap_amount: int,
    eth_payout_address: str,
    expired_swap_reservation_indexes: list[int],
    rift_exchange: Contract,
    private_key: str,
    w3: Web3,
) -> int:
    account = w3.eth.account.from_key(private_key).address

    id = await get_reservation_vaults_length(rift_exchange)
    await build_and_send_transaction(
        w3,
        account,
        rift_exchange.functions.reserveLiquidity(
            vaultIndexesToReserve=vault_indexes_to_reserve,
            amountsToReserve=amounts_to_reserve,
            ethPayoutAddress=eth_payout_address,
            expiredSwapReservationIndexes=expired_swap_reservation_indexes,
        ),
        private_key,
    )
    return id

async def deposit_liquidity_rift(
    btc_payout_locking_script: str,
    btc_exchange_rate: int,
    vault_index_to_overwrite: int,
    deposit_amount: int,
    vault_index_with_same_exchange_rate: int,
    rift_exchange: Contract,
    private_key: str,
    w3: Web3,
):
    account = w3.eth.account.from_key(private_key).address
    await build_and_send_transaction(
        w3,
        account,
        rift_exchange.functions.depositLiquidity(
            btcPayoutLockingScript=bytes.fromhex(normalize_hex_str(btc_payout_locking_script)),
            exchangeRate=btc_exchange_rate,
            vaultIndexToOverwrite=vault_index_to_overwrite,
            depositAmount=deposit_amount,
            vaultIndexWithSameExchangeRate=vault_index_with_same_exchange_rate,
        ),
        private_key,
    )



async def get_deposit_vaults_length(rift_exchange: Contract):
    return await asyncio.to_thread(rift_exchange.functions.getDepositVaultsLength().call)

async def get_reservation_vaults_length(rift_exchange: Contract):
    return await asyncio.to_thread(rift_exchange.functions.getReservationLength().call)


async def get_reservation(reservation_index: int, rift_exchange: Contract) -> SwapReservation:
    raw_data = await asyncio.to_thread(rift_exchange.functions.getReservation(reservation_index).call)
    return SwapReservation(
        confirmation_block_height=raw_data[0],
        reservation_timestamp=raw_data[1],
        unlock_timestamp=raw_data[2],
        state=ReservationState(raw_data[3]),
        eth_payout_address=Address(raw_data[4]),
        lp_reservation_hash=HexStr(raw_data[5].hex()),
        nonce=HexStr(raw_data[6].hex()),
        total_swap_amount=raw_data[7],
        prepaid_fee_amount=raw_data[8],
        vault_indexes=raw_data[9],
        amounts_to_reserve=raw_data[10]
    )


@testnet_autoexit
async def main(anvil: AnvilInstance | None = None):
    assert isinstance(anvil, AnvilInstance)
    private_key = hashlib.sha256(b"rift-test").hexdigest()
    swap_txid = "8df99d697780166f12df68b1e2410a909374b6414da57a1a65f3b84eb8a4dd0f"
    proposed_height = 854136
    safe_height = 854133
    safe_hash = "000000000000000000022e9a6fdb38378989170359d2afbb47bee7e251daf9fd"
    w3: AnvilWeb3 = AnvilWeb3(HTTPProvider(anvil.http_url))
    address = w3.eth.account.from_key(private_key).address

    ETH_BTC_RATE = 20.5
    wei_sats_rate = int(ETH_BTC_RATE * 10**18) // 10**8
    print("ETH/BTC Rate:", ETH_BTC_RATE, "ether per 1 bitcoin")
    print("Wei/Sats Rate:", wei_sats_rate)
    per_lp_amount = int((1)*10**18)
    lps = [
        LiquidityProvider(
            amount=per_lp_amount,
            btc_exchange_rate=wei_sats_rate,
            locking_script_hex="001463dff5f8da08ca226ba01f59722c62ad9b9b3eaa",
        ),
        LiquidityProvider(
            amount=per_lp_amount,
            btc_exchange_rate=wei_sats_rate,
            locking_script_hex="0014aa86191235be8883693452cf30daf854035b085b",
        ),
        LiquidityProvider(
            amount=per_lp_amount,
            btc_exchange_rate=wei_sats_rate,
            locking_script_hex="00146ab8f6c80b8a7dc1b90f7deb80e9b59ae16b7a5a",
        ),
    ]
    # BTC/ETH is 1000
    # amount of ETH being deposited is 0.0001
    # thus the amount of BTC expected is 

    await fund_account_with_ether(private_key, w3)
    weth = await deploy_weth(private_key, w3)
    rift_exchange = await deploy_rift_exchange(
        initial_checkpoint_height=safe_height,
        initial_block_hash=safe_hash,
        verifier_contract_address=(await deploy_verifier(private_key, w3)).address,
        deposit_token_address=weth.address,
        private_key=private_key,
        w3=w3,
    )

    await deposit_and_approve_weth(
        weth=weth,
        spender=rift_exchange.address,
        amount=10**18 * 100,
        private_key=private_key,
        w3=w3,
    )

    for lp in lps:
        await deposit_liquidity_rift(
            btc_payout_locking_script=lp.locking_script_hex,
            btc_exchange_rate=lp.btc_exchange_rate,
            vault_index_to_overwrite=-1,
            deposit_amount=lp.amount,
            vault_index_with_same_exchange_rate=-1,
            rift_exchange=rift_exchange,
            private_key=private_key,
            w3=w3,
        )


    

    utilized_lps = [
        LiquidityProvider(
            amount=w3.to_wei(0.0001, "ether"),
            btc_exchange_rate=wei_sats_rate,
            locking_script_hex="001463dff5f8da08ca226ba01f59722c62ad9b9b3eaa",
        ),
        LiquidityProvider(
            amount=w3.to_wei(0.0001, "ether"),
            btc_exchange_rate=wei_sats_rate,
            locking_script_hex="0014aa86191235be8883693452cf30daf854035b085b",
        ),
        LiquidityProvider(
            amount=w3.to_wei(0.0001, "ether"),
            btc_exchange_rate=wei_sats_rate,
            locking_script_hex="00146ab8f6c80b8a7dc1b90f7deb80e9b59ae16b7a5a",
        ),
    ]
    print("Expected Sats per LP")
    for i, lp in enumerate(utilized_lps):
        print(f"LP #{i}:", wei_to_satoshi(lp.amount, wei_sats_rate), "sats")

    initial_total_swap_amount = sum(map(lambda lp: lp.amount, utilized_lps))
    fees = calculate_total_fees_wei(
        totalSwapAmount=initial_total_swap_amount,
        basefee=await get_basefee_wei(w3),
    )
    total_swap_amount = initial_total_swap_amount + fees

    # necesary to account for rounding errors as a result of integer only math
    true_payout = sum(wei_to_satoshi(lp.amount, lp.btc_exchange_rate) * lp.btc_exchange_rate for lp in utilized_lps)

    print("Fees as percentage of total swap amount", round((fees / (initial_total_swap_amount + fees))*100, 2))
    print("Total USD Swap Amount", round((total_swap_amount/(10**18))*3400, 2))
    print("Total ETH Swap Amount", total_swap_amount / 10**18, "ether")
    print("Total BTC Being Sent", wei_to_satoshi(sum(map(lambda lp: lp.amount, utilized_lps)), wei_sats_rate), "sats")
    print("Total ETH Being Swapped", initial_total_swap_amount / 10**18, "ether")
    
    # something constant in the future so the order nonce is constant
    w3.anvil.set_next_block_timestamp(2524640461)

    reservation_id = await reserve_liquidity_rift(
        vault_indexes_to_reserve=[0, 1, 2],
        amounts_to_reserve=list(map(lambda lp: lp.amount, utilized_lps)),
        total_swap_amount=true_payout,
        eth_payout_address=address,
        expired_swap_reservation_indexes=[],
        rift_exchange=rift_exchange,
        private_key=private_key,
        w3=w3
    )

    order_nonce = (await get_reservation(reservation_id, rift_exchange)).nonce
    print("Order nonce", order_nonce)
    #print("LPS", utilized_lps)

    rift_bitcoin_data = await get_rift_btc_data(
        proposed_block_height=proposed_height,
        safe_block_height=safe_height,
        txid=swap_txid,
        mainnet=True
    )


    async with change_dir_async('circuits'):
        giga_proof_artifact = await build_giga_circuit_proof_and_input(
            txn_data_no_segwit_hex=rift_bitcoin_data.txn_data_no_segwit_hex,
            lp_reservations=utilized_lps,
            proposed_block_header=rift_bitcoin_data.proposed_block_header,
            safe_block_header=rift_bitcoin_data.safe_block_header,
            retarget_block_header=rift_bitcoin_data.retarget_block_header,
            inner_block_headers=rift_bitcoin_data.inner_block_headers,
            confirmation_block_headers=rift_bitcoin_data.confirmation_block_headers,
            order_nonce_hex=order_nonce,
            expected_payout=true_payout,
            safe_block_height=safe_height,
            block_height_delta=rift_bitcoin_data.block_height_delta
        )

    print("Giga Proof Artifact", giga_proof_artifact)
    

        



    



if __name__ == "__main__":
    asyncio.run(main())
