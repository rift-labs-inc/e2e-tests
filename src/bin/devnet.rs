use test_utils::spawn; 

#[tokio::main]
async fn main() {
    let mut client = Client::new("devnet.rs").await.unwrap();
    let block = client.get_block(1).await.unwrap();
    println!("{:?}", block);
}
