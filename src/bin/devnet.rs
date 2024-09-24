use test_utils::core::RiftDevnet; 
use eyre::Result;
use tokio::signal;

#[tokio::main]
async fn main() -> Result<()> {
    let devnet = RiftDevnet::setup().await?;
    signal::ctrl_c().await?;
    drop(devnet);
    Ok(())
}
