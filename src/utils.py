from pydantic import BaseModel, Field
import asyncio
from typing import List, Dict, Any, Optional
from enum import Enum
from eth_typing import Address, HexStr

class ReservationState(Enum):
    None_ = 0
    Created = 1
    Unlocked = 2
    ExpiredAndAddedBackToVault = 3
    Completed = 4


"""
struct SwapReservation {
    uint32 confirmationBlockHeight;
    uint32 reservationTimestamp;
    uint32 unlockTimestamp; // timestamp when reservation was proven and unlocked
    ReservationState state;
    address ethPayoutAddress;
    bytes32 lpReservationHash;
    bytes32 nonce; // sent in bitcoin tx calldata from buyer -> lps to prevent replay attacks
    uint256 totalSwapAmount;
    int256 prepaidFeeAmount;
    uint256 proposedBlockHeight;
    bytes32 proposedBlockHash;
    uint256[] vaultIndexes;
    uint192[] amountsToReserve;
}
"""

class SwapReservation(BaseModel):
    confirmation_block_height: int
    reservation_timestamp: int
    unlock_timestamp: int
    state: ReservationState
    eth_payout_address: Address
    lp_reservation_hash: HexStr
    nonce: HexStr
    total_swap_amount: int
    prepaid_fee_amount: int
    proposed_block_height: int
    proposed_block_hash: HexStr
    vault_indexes: List[int]
    amounts_to_reserve: List[int]


class ABIItem(BaseModel):
    type: str
    name: Optional[str] = None
    inputs: Optional[List[Dict[str, Any]]] = None
    outputs: Optional[List[Dict[str, Any]]] = None
    stateMutability: Optional[str] = None

class BytecodeObject(BaseModel):
    object: str
    sourceMap: Optional[str] = None
    linkReferences: Dict[str, Any] = {}
    immutableReferences: Optional[Dict[str, List[Dict[str, int]]]] = None

class StorageLayoutType(BaseModel):
    encoding: str
    label: str
    numberOfBytes: str
    base: Optional[str] = None
    members: Optional[List[Dict[str, Any]]] = None
    key: Optional[str] = None
    value: Optional[str] = None

class StorageLayout(BaseModel):
    storage: List[Dict[str, Any]]
    types: Dict[str, StorageLayoutType]

class ContractArtifact(BaseModel):
    abi: list[Any] 
    bytecode: BytecodeObject
    deployedBytecode: BytecodeObject
    methodIdentifiers: Dict[str, str]
    rawMetadata: str
    metadata: Dict[str, Any]
    storageLayout: Optional[StorageLayout] = None
    id: int

    class Config:
        extra = 'allow'  # This allows extra fields that are not defined in the model

class LiquidityProvider(BaseModel):
    amount: int
    btc_exchange_rate: int
    locking_script_hex: str


async def run_command(command: str, cwd: str, strict_failure = True) -> bytes:
    process = await asyncio.create_subprocess_shell(
        command,
        shell=True,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    stdout, stderr = await process.communicate()
    if strict_failure:
        if process.returncode != 0 or stderr:
            raise Exception(f"`{command}` failed with {stderr.decode().strip()}")
    else:
        if process.returncode != 0:
            raise Exception(f"`{command}` failed with {stderr.decode().strip()}")
    return stdout

def normalize_hex_str(hex_str: str) -> str:
    mod_str = hex_str
    if hex_str.startswith("0x"):
        mod_str = hex_str[2:]
    if len(hex_str) % 2 != 0:
        mod_str = f"0{mod_str}"
    return mod_str

def wei_to_satoshi(wei_amount: int, wei_sats_exchange_rate: int):
    return wei_amount // wei_sats_exchange_rate

def sats_to_wei(sats_amount: int, wei_sats_exchange_rate: int):
    return sats_amount * wei_sats_exchange_rate
