use alloy::primitives::U256;
use alloy::sol;
use bitcoin::consensus::serde::hex::Lower;
use bitcoin::hex::DisplayHex;
use bitcoin::{Amount, Script, Txid};
use std::collections::HashMap;
use std::time::Duration;
use std::{str::FromStr, sync::Arc};
use tokio::time::sleep;

use bitcoin::{
    address::NetworkChecked,
    hashes::{serde::Serialize, Hash},
    Address, PublicKey,
};
// Spawn bitcoin and anvil processes, and deploy contracts to anvil.
use alloy::{
    hex,
    network::EthereumWallet,
    node_bindings::{Anvil, AnvilInstance},
    providers::{ProviderBuilder, WsConnect},
    signers::local::PrivateKeySigner,
};
use bitcoind::{bitcoincore_rpc::RpcApi, BitcoinD};
use eyre::{eyre, Result};
use hypernode::btc_rpc::BitcoinRpcClient;
use hypernode::{
    self,
    core::{EvmWebsocketProvider, RiftExchange, RiftExchangeWebsocket},
};
use rift_lib::transaction::P2WPKHBitcoinWallet;

use bitcoind::bitcoincore_rpc::{self, Client};
use bitcoind::bitcoincore_rpc::json::ImportDescriptors;

fn get_new_core_aware_address(
    bitcoin_regtest_instance: &BitcoinD,
    network: bitcoin::Network,
) -> Address<NetworkChecked> {
    // First, try to get an existing address from the wallet
    let new_address = bitcoin_regtest_instance
        .client
        .get_new_address(
            None,
            Some(bitcoind::bitcoincore_rpc::json::AddressType::Bech32),
        )
        .unwrap();
    new_address.require_network(network).unwrap()
}

pub struct RiftDevnet {
    pub bitcoin_regtest_instance: BitcoinD,
    pub anvil_instance: AnvilInstance,
    pub funded_btc_wallet: P2WPKHBitcoinWallet,
    // pub contract: Arc<hypernode::core::RiftExchangeWebsocket>
}

impl RiftDevnet {

    pub async fn setup() -> Result<Self> {
        let network = bitcoin::Network::Regtest;
        let (bitcoin_regtest, anvil) = tokio::try_join!(spawn_bitcoin_regtest(), spawn_anvil())?;

        println!("Bitcoin Regtest Url: {}", bitcoin_regtest.rpc_url());
        println!(
            "Bitcoin Regtest Cookie File: {}",
            bitcoin_regtest.params.cookie_file.display()
        );
        println!("Anvil Url: {}", anvil.endpoint());

        println!(
            "bitcoin-cli connect: \"bitcoin-cli -regtest -rpcport={} -rpccookiefile={}\"",
            bitcoin_regtest.params.rpc_socket.port(),
            bitcoin_regtest.params.cookie_file.display()
        );

        // Constant miner address for test consistency 
        let private_key = hex!("000000000000000000000000000000000000000000000000000000000000dead");

        let miner = get_new_core_aware_address(&bitcoin_regtest, network);

        let funded_btc_wallet = P2WPKHBitcoinWallet::from_secret_key(private_key, network);

        // Generate blocks to the miner's address
        bitcoin_regtest
            .client
            .generate_to_address(120, &miner)
            .unwrap();

        let txid = bitcoin_regtest.client.send_to_address(
            &funded_btc_wallet.address,
            Amount::from_btc(1000.0).unwrap(),
            None,
            None,
            Some(true),
            None,
            None,
            None,
        ).unwrap();

        println!("txid: {:?}", txid);

        // mine the tx
        bitcoin_regtest
            .client
            .generate_to_address(6, &miner)
            .unwrap();

        Ok(RiftDevnet {
            bitcoin_regtest_instance: bitcoin_regtest,
            anvil_instance: anvil,
            funded_btc_wallet,
        })
    }
}

sol!(
    
    );

async fn deploy_sp1_verifier(anvil: &AnvilInstance) -> Result<Address<NetworkChecked>> {
    let signer: PrivateKeySigner = anvil.keys()[0].clone().into();
    let wallet = EthereumWallet::from(signer);
    let provider: EvmWebsocketProvider =
    ProviderBuilder::new()
        .with_recommended_fillers()
        .wallet(wallet)
        .on_ws(WsConnect::new(anvil.ws_endpoint_url()))
        .await
        .expect("Failed to connect to WebSocket");

    let verifier_contract = Verifier::deploy(provider).await?;
    Ok(verifier_contract.address())
}


async fn deploy_contract(anvil: &AnvilInstance, bitcoind_client: &Client) -> Result<Arc<RiftExchangeWebsocket>> {
    let signer: PrivateKeySigner = anvil.keys()[0].clone().into();
    let wallet = EthereumWallet::from(signer);
    let provider: EvmWebsocketProvider =
    ProviderBuilder::new()
        .with_recommended_fillers()
        .wallet(wallet)
        .on_ws(WsConnect::new(anvil.ws_endpoint_url()))
        .await
        .expect("Failed to connect to WebSocket");

    let initial_checkpoint_height = bitcoind_client.get_block_count()?;
    let initial_block_hash = bitcoind_client.get_block_hash(initial_checkpoint_height)?;
    let retarget_block_height = initial_checkpoint_height - (initial_checkpoint_height % 2016); 
    let initial_retarget_block_hash = bitcoind_client.get_block_hash(retarget_block_height)?;

    println!("Initial block hash: {}", initial_block_hash);

    let contract = RiftExchange::deploy(
        provider,
        U256::from(initial_checkpoint_height),
        initial_block_hash.to_byte_array().into(),
        initial_retarget_block_hash.to_byte_array().into(),
        verifierContractAddress,
        depositTokenAddress,
        proverReward,
        releaserReward,
        protocolAddress, owner, circuitVerificationKey, minimumConfirmationDelta)
    Ok(Arc::new(contract))
}

async fn spawn_bitcoin_regtest() -> Result<BitcoinD> {
    tokio::task::spawn_blocking(|| {
        BitcoinD::new(bitcoind::exe_path().map_err(|e| eyre!(e))?).map_err(|e| eyre!(e))
    })
    .await?
}

async fn spawn_anvil() -> Result<AnvilInstance> {
    tokio::task::spawn_blocking(|| {
        Anvil::new()
            .block_time(1)
            .chain_id(1337)
            .try_spawn()
            .map_err(|e| eyre!(e))
    })
    .await?
}
