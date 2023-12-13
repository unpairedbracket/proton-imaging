use std::{thread, time::Duration};

use async_channel::Sender;

async fn produce_edges(edges: impl Iterator<Item = u8>, chan: Sender<u8>) {
    for edge in edges {
        println!("sending {edge}");
        match chan.send(edge).await {
            Ok(_) => (),
            Err(_) => break,
        }
    }
}

#[tokio::main(worker_threads = 4)]
async fn test_producer() {
    let n = 4;
    let numbers = 0..20;

    let (send, recv) = async_channel::bounded(n);

    let mut recvs = vec![];
    for _ in 0..n {
        let local_recv = recv.clone();
        recvs.push(tokio::spawn(async move {
            while let Ok(num) = local_recv.recv().await {
                println!("Got {num}");
                thread::sleep(Duration::from_secs(1));
            }
        }));
    }

    drop(recv);

    let prod = tokio::spawn(produce_edges(numbers, send));

    prod.await.unwrap();
    for r in recvs {
        r.await.unwrap();
    }
}

#[test]
fn test() {
    test_producer()
}
