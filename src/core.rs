use crate::constants::{MINIMUM_CONFIRMATION_DELTA, PROVER_REWARD_USD, RELEASER_REWARD_USD};
use chrono;
use crate::sp1_core::{SP1_CIRCUIT_VERIFICATION_HASH, SP1_VERIFIER_BYTECODE};
use alloy::hex::FromHex;
use alloy::network::NetworkWallet;
use alloy::primitives::U256;
use alloy::providers::ext::AnvilApi;
use alloy::providers::{Provider, WalletProvider};
use alloy::sol;
use bitcoin::consensus::serde::hex::Lower;
use bitcoin::hex::DisplayHex;
use bitcoin::{Amount, Script, Txid};
use hypernode::{btc_rpc, HypernodeArgs};
use std::collections::HashMap;
use std::ops::Mul;
use std::thread;
use std::time::Duration;
use std::{str::FromStr, sync::Arc};
use tokio::runtime::Runtime;
use tokio::time::sleep;

use bitcoin::{
    address::NetworkChecked,
    hashes::{serde::Serialize, Hash},
    PublicKey,
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
use hypernode::{
    self,
    core::{EvmWebsocketProvider, RiftExchange, RiftExchangeWebsocket},
};
use rift_lib::transaction::P2WPKHBitcoinWallet;

use bitcoind::bitcoincore_rpc::Client;

fn get_new_core_aware_address(
    bitcoin_regtest_instance: &BitcoinD,
    network: bitcoin::Network,
) -> bitcoin::Address<NetworkChecked> {
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
    pub deposit_token: alloy::primitives::Address,
    pub contract: Arc<hypernode::core::RiftExchangeWebsocket>,
    evm_ws_rpc: String,
    btc_rpc: String,
    hypernode_signer: PrivateKeySigner
}

impl RiftDevnet {
    pub async fn run_hypernode(&self) -> Result<()> {
        let rift_exchange_address = self.contract.address().to_string();
        let rpc_concurrency = 10;
        let btc_polling_interval = 2;
        let mock_proof = false;
        let flashbots = false;
        let flashbots_relay_rpc: Option<String> = None;
        let private_key = self.hypernode_signer.clone().into_credential().to_bytes().to_lower_hex_string();
        let evm_ws_rpc = self.evm_ws_rpc.clone();
        let btc_rpc = self.btc_rpc.clone();
        // spawn hypernode
        thread::spawn(move || {
            let runtime = Runtime::new().expect("Failed to create Tokio runtime");
            runtime.block_on(async {
                hypernode::node::run(HypernodeArgs {
                    evm_ws_rpc,
                    btc_rpc,
                    private_key,
                    rift_exchange_address,
                    rpc_concurrency,
                    btc_polling_interval,
                    mock_proof,
                    flashbots,
                    flashbots_relay_rpc,
                })
                .await
                .expect("Hypernode run failed");
            });
        });
        println!("Setup complete...");
        Ok(())
    }

    pub async fn setup() -> Result<Self> {
        let network = bitcoin::Network::Regtest;
        let (bitcoin_regtest, anvil) = tokio::try_join!(spawn_bitcoin_regtest(), spawn_anvil())?;

        let cookie = bitcoin_regtest.params.get_cookie_values()?.unwrap();
        let btc_rpc = format!(
            "http://{}:{}@127.0.0.1:{}",
            cookie.user,
            cookie.password,
            bitcoin_regtest.params.rpc_socket.port()
        );

        println!("---RIFT DEVNET---");

        println!("Bitcoin Regtest Url: {}", btc_rpc);

        println!(
            "bitcoin-cli connect: \"bitcoin-cli -regtest -rpcport={} -rpccookiefile={}\"",
            bitcoin_regtest.params.rpc_socket.port(),
            bitcoin_regtest.params.cookie_file.display()
        );

        println!("Anvil Url: {}", anvil.endpoint());

        // Constant miner address for test consistency
        let private_key = hex!("000000000000000000000000000000000000000000000000000000000000dead");

        let miner = get_new_core_aware_address(&bitcoin_regtest, network);

        let funded_btc_wallet = P2WPKHBitcoinWallet::from_secret_key(private_key, network);

        // Generate blocks to the miner's address
        bitcoin_regtest
            .client
            .generate_to_address(120, &miner)
            .unwrap();

        let _txid = bitcoin_regtest
            .client
            .send_to_address(
                &funded_btc_wallet.address,
                Amount::from_btc(1000.0).unwrap(),
                None,
                None,
                Some(true),
                None,
                None,
                None,
            )
            .unwrap();

        // mine the tx
        bitcoin_regtest
            .client
            .generate_to_address(6, &miner)
            .unwrap();

        // now setup contracts
        let (rift_exchange, deposit_token) =
            deploy_contracts(&anvil, &bitcoin_regtest.client).await?;

        let provider = rift_exchange.provider().clone();

        let evm_ws_rpc: String = anvil.ws_endpoint();

        let hypernode_signer: PrivateKeySigner = anvil.keys()[1].clone().into();
        let hypernode_address = hypernode_signer.address();

        // give some eth using anvil
        provider
            .anvil_set_balance(hypernode_address, U256::from_str("10000000000000000000")?)
            .await?;

        
        Ok(RiftDevnet {
            bitcoin_regtest_instance: bitcoin_regtest,
            anvil_instance: anvil,
            funded_btc_wallet,
            contract: rift_exchange,
            deposit_token,
            evm_ws_rpc,
            btc_rpc,
            hypernode_signer
        })
    }
}

sol!(
    #[allow(missing_docs)]
    #[sol(rpc)]
    MockUSDT,
    "hypernode/artifacts/MockUSDT.json"
);

async fn deploy_contracts(
    anvil: &AnvilInstance,
    bitcoind_client: &Client,
) -> Result<(Arc<RiftExchangeWebsocket>, alloy::primitives::Address)> {
    let signer: PrivateKeySigner = anvil.keys()[0].clone().into();
    println!("Exchange owner address: {}", signer.address());
    let wallet = EthereumWallet::from(signer.clone());
    let provider: Arc<EvmWebsocketProvider> = Arc::new(
        ProviderBuilder::new()
            .with_recommended_fillers()
            .wallet(wallet)
            .on_ws(WsConnect::new(anvil.ws_endpoint_url()))
            .await
            .expect("Failed to connect to WebSocket"),
    );
    let verifier_contract =
        alloy::primitives::Address::from_str("0xaeE21CeadF7A03b3034DAE4f190bFE5F861b6ebf")?;
    provider
        .anvil_set_code(
            verifier_contract,
            Vec::from_hex(SP1_VERIFIER_BYTECODE)?.into(),
        )
        .await?;
    let usdt_contract = MockUSDT::deploy(provider.clone()).await?;
    let initial_checkpoint_height = bitcoind_client.get_block_count()?;
    let initial_block_hash = bitcoind_client.get_block_hash(initial_checkpoint_height)?;
    let initial_block_chainwork: [u8; 32] = bitcoind_client
        .get_block_header_info(&initial_block_hash)?
        .chainwork
        .try_into()
        .unwrap();
    let retarget_block_height = initial_checkpoint_height - (initial_checkpoint_height % 2016);
    let initial_retarget_block_hash = bitcoind_client.get_block_hash(retarget_block_height)?;
    println!("USDT address: {}", usdt_contract.address());

    let contract = RiftExchange::deploy(
        provider.clone(),
        U256::from(initial_checkpoint_height),
        initial_block_hash.to_byte_array().into(),
        initial_retarget_block_hash.to_byte_array().into(),
        U256::from_be_bytes(initial_block_chainwork),
        verifier_contract,
        *usdt_contract.address(),
        U256::from(PROVER_REWARD_USD.mul(10_u64.pow(6_u32))),
        U256::from(RELEASER_REWARD_USD.mul(10_u64.pow(6_u32))),
        signer.address(),
        signer.address(),
        SP1_CIRCUIT_VERIFICATION_HASH.into(),
        MINIMUM_CONFIRMATION_DELTA,
    )
    .await?;

    println!("Exchange address: {}", contract.address());

    Ok((Arc::new(contract), *usdt_contract.address()))
}

async fn spawn_bitcoin_regtest() -> Result<BitcoinD> {
    tokio::task::spawn_blocking(|| {
        BitcoinD::new(bitcoind::exe_path().map_err(|e| eyre!(e))?).map_err(|e| eyre!(e))
    })
    .await?
}

async fn spawn_anvil() -> Result<AnvilInstance> {
    tokio::task::spawn_blocking(|| {

        let _ = Anvil::new().arg("--accounts").arg("20").spawn();
        Anvil::new()
            .block_time(1)
            .chain_id(1337)
            .arg("--timestamp")
            .arg((chrono::Utc::now().timestamp() - 9 * 60 * 60).to_string())
            .try_spawn()
            .map_err(|e| eyre!(e))
    })
    .await?
}
