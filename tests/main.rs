
#[cfg(test)]
mod integration_tests {
    use std::str::FromStr;

    use alloy::primitives::{Address, U256};
    use tokio;
    use test_utils::core::RiftDevnet;
    use eyre::Result;
    use alloy::providers::ext::AnvilApi;

    async fn setup() -> Result<RiftDevnet> {
        RiftDevnet::setup().await
    }

    async fn teardown(devnet: RiftDevnet) {
        drop(devnet);
    }

    #[tokio::test]
    async fn test_basic_swap() -> Result<()> {
        let devnet = setup().await?;
        devnet.run_hypernode().await?;

        slot = w3.keccak(encode(["address", "uint256"], [signer.address, 9]))
        let signer_address = Address::from_str("0x742d35Cc6634C0532925a3b844Bc454e4438f44e").unwrap();
        
        // The uint256 value
        let value = U256::from(9);

        // Encode the parameters
        let encoded = (signer_address, value).abi_encode();

        w3.anvil.set_storage_at(USDC, int.from_bytes(slot), encode(["uint256"], [10000]))

        devnet.contract.provider().anvil_set_storage_at(
            devnet.contract.address(),
            U256::from(0),
            U256::from(1000000000000000000),
        ).await?;

        // deposit 100 usd
        devnet.contract.depositLiquidity(U256::from(), ).await?;
        

        devnet.contract.reserveLiquidity(vaultIndexesToReserve, amountsToReserve, ethPayoutAddress, totalSatsInputInlcudingProxyFee, expiredSwapReservationIndexes)

        rift_lib::transaction::build_rift_payment_transaction(order_nonce, liquidity_providers, in_txid, transaction, in_txvout, wallet, fee_sats)

        // broadcast using btc rpc

        // ... hypernode picks it up automatically


        // Your test code here
        // For example:
        // let result = e2e_tests::some_async_function().await?;
        // assert_eq!(result, expected_value);

        teardown(devnet).await;
        Ok(())
    }
}
