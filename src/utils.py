from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum
from eth_typing import Address, HexStr

class ReservationState(Enum):
    None_ = 0
    Created = 1
    Unlocked = 2
    ExpiredAndAddedBackToVault = 3
    Completed = 4

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
    storageLayout: StorageLayout
    id: int

    class Config:
        extra = 'allow'  # This allows extra fields that are not defined in the model



