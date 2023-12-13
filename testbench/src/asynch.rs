use std::iter;

use async_scoped::TokioScope;
use crossbeam::deque::{Injector, Worker};
use ndarray::{Array, ArrayViewMut2};

use crate::acc::{self, Edge, Point};

const N_THREADS: usize = 4;

#[tokio::main(worker_threads = 8)]
pub async fn _compute_async(
    edges: impl Iterator<Item = Edge> + Send,
    mut raster: ArrayViewMut2<f64>,
    xy_min: Point<isize>,
) {
    let sh = raster.raw_dim();

    let (send, recv) = async_channel::bounded(N_THREADS);

    let ((), results) = TokioScope::scope_and_block(|scope| {
        for _ in 0..N_THREADS {
            let local_recv = recv.clone();
            scope.spawn(async move {
                let mut temp_raster = Array::zeros(sh);
                while let Ok(edge) = local_recv.recv().await {
                    acc::fill_accumulation(&mut temp_raster, &edge, &xy_min)
                }
                Some(temp_raster)
            });
        }

        drop(recv);

        scope.spawn(async move {
            for edge in edges {
                match send.send(edge).await {
                    Ok(_) => (),
                    Err(_) => break,
                }
            }
            None
        });
    });

    for r in results {
        if let Some(a) = r.unwrap() {
            raster += &a;
        }
    }
}

pub fn _compute_crossbeam(
    edges: impl Iterator<Item = Edge> + Send,
    mut raster: ArrayViewMut2<f64>,
    xy_min: Point<isize>,
) {
    let sh = raster.raw_dim();

    let (send, recv) = crossbeam::channel::bounded(N_THREADS);

    let _ = crossbeam::scope(|scope| {
        let mut handles = vec![];
        for _ in 0..N_THREADS {
            let local_recv = recv.clone();
            handles.push(scope.spawn(move |_| {
                let mut temp_raster = Array::zeros(sh);
                while let Ok(edge) = local_recv.recv() {
                    acc::fill_accumulation(&mut temp_raster, &edge, &xy_min);
                }
                temp_raster
            }));
        }

        drop(recv);

        scope.spawn(move |_| {
            for edge in edges {
                match send.send(edge) {
                    Ok(_) => (),
                    Err(_) => break,
                }
            }
        });

        for r in handles {
            let a = r.join().unwrap();
            raster += &a;
        }
    });
}

pub fn compute_steal(
    edges: impl Iterator<Item = Edge> + Send,
    mut raster: ArrayViewMut2<f64>,
    xy_min: Point<isize>,
) {
    let sh = raster.raw_dim();

    let injector = Injector::new();
    let workers = Vec::from_iter((0..N_THREADS).map(|_| Worker::new_fifo()));
    let stealers = Vec::from_iter(workers.iter().map(|w| w.stealer()));

    let _ = crossbeam::scope(|scope| {
        let mut handles = vec![];

        for local in workers {
            let stealers = &stealers;
            let global = &injector;
            handles.push(scope.spawn(move |_| {
                let mut temp_raster = Array::zeros(sh);
                loop {
                    let Some(edge) = local.pop().or_else(|| {
                        // Otherwise, we need to look for a task elsewhere.
                        iter::repeat_with(|| {
                            // Try stealing a batch of tasks from the global queue.
                            global.steal_batch_and_pop(&local)
                            // Or try stealing a task from one of the other threads.
                            .or_else(|| stealers.iter().map(|s| s.steal()).collect())
                        })
                        // Loop while no task was stolen and any steal operation needs to be retried.
                        .find(|s| !s.is_retry())
                        // Extract the stolen task, if there is one.
                        .and_then(|s| s.success())
                    }) else {break};
                    acc::fill_accumulation(&mut temp_raster, &edge, &xy_min);
                }
                temp_raster
            }));
        }

        for edge in edges {
            injector.push(edge);
        }
        println!("finished pushing");

        // let global = &injector;
        // scope.spawn(move |_| {
        //     for edge in edges {
        //         global.push(edge);
        //     }
        //     println!("finished pushing");
        // });

        for r in handles {
            let a = r.join().unwrap();
            raster += &a;
        }
    });
}
