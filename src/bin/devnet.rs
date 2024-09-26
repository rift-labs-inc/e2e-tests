use test_utils::core::RiftDevnet; 
use eyre::Result;
use tokio::signal;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let devnet = RiftDevnet::setup().await?;
    devnet.run_hypernode().await?;
    signal::ctrl_c().await?;
    drop(devnet);
    Ok(())
}
