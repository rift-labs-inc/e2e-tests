// Spawn bitcoin and anvil processes, and deploy contracts to anvil.
use eyre::Result;
use alloy::node_bindings::Anvil;
use alloy::sol;

sol!(

    );

pub async fn spawn_bitcoin_regtest() -> Result<bitcoind::BitcoinD> {
    tokio::task::spawn_blocking(|| {
        Ok(bitcoind::BitcoinD::new(bitcoind::exe_path().unwrap()).unwrap())
    })
    .await?
}

pub async fn spawn_anvil() -> Result<anvil::Anvil> {
    tokio::task::spawn_blocking(|| {
        Anvil::new().block_time(1).chain_id(1337).try_spawn()?
    })
    .await?
}

pub async fn bootstrap_devnet() -> Result<(bitcoind::BitcoinD, anvil::Anvil)> {
    let bitcoin = spawn_bitcoin_regtest().await?;
    let anvil = spawn_anvil().await?;
    Ok((bitcoin, anvil))
} 

