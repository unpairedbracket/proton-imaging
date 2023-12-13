mod acc;
mod asynch;
pub mod qt;

use acc::{fill_accumulation, Edge, Point};
use asynch::compute_steal;
use ndarray::{Array2, ArrayViewMut2, Axis};
use numpy::{PyReadonlyArray2, PyReadwriteArray2};
use pyo3::prelude::*;
use rayon::{current_num_threads, current_thread_index, prelude::*};

#[pyfunction]
fn fill_accumulation_fold(
    mut raster: PyReadwriteArray2<f64>,
    edges: PyReadonlyArray2<f64>,
    xy_min: (isize, isize),
) -> PyResult<()> {
    let mut ras = raster.as_array_mut();
    let sh = ras.raw_dim();
    let _chunk_size =
        1000.max((edges.shape()[0] + current_num_threads() - 1) / current_num_threads());
    // println!(
    //     "{:?} edges on {} threads -> at most {} per thread",
    //     edges.shape()[0],
    //     current_num_threads(),
    //     chunk_size
    // );
    let xy_min = Point {
        x: xy_min.0,
        y: xy_min.1,
    };

    let edges = edges.as_array();

    let piter = edges.axis_iter(Axis(0)).into_par_iter(); //par_bridge();
    let solution = piter
        .fold(
            || {
                println!("running id on {:?}", current_thread_index());
                Array2::zeros(sh)
            },
            |mut im, edge| {
                let edge = Edge::new(
                    Point {
                        x: edge[0],
                        y: edge[1],
                    },
                    Point {
                        x: edge[2],
                        y: edge[3],
                    },
                    edge[4],
                );
                fill_accumulation(&mut im, &edge.try_into().unwrap(), &xy_min);
                im
            },
        )
        .reduce(
            || {
                println!("id for reduce on {:?}", current_thread_index());
                Array2::zeros(sh)
            },
            |im0, im1| im0 + im1,
        );

    ras += &solution;
    Ok(())
}

#[pyfunction]
fn fill_accumulation_fold2(
    mut raster: PyReadwriteArray2<f64>,
    edges: PyReadonlyArray2<f64>,
    xy_min: (isize, isize),
) -> PyResult<()> {
    let mut ras = raster.as_array_mut();

    let xy_min = Point {
        x: xy_min.0,
        y: xy_min.1,
    };

    let edges = edges.as_array();
    let edges = edges.axis_iter(Axis(0)).map(|edge| {
        Edge::new(
            Point {
                x: edge[0],
                y: edge[1],
            },
            Point {
                x: edge[2],
                y: edge[3],
            },
            edge[4],
        )
    });

    fill(&mut ras, edges, &xy_min);

    Ok(())
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn fill_accumulation_async(
    mut raster: PyReadwriteArray2<f64>,
    edges: PyReadonlyArray2<f64>,
    xy_min: (isize, isize),
) -> PyResult<()> {
    let ras = raster.as_array_mut();

    let xy_min = Point {
        x: xy_min.0,
        y: xy_min.1,
    };

    let edges = edges.as_array();
    let edges = edges.axis_iter(Axis(0)).map(|edge| {
        Edge::new(
            Point {
                x: edge[0],
                y: edge[1],
            },
            Point {
                x: edge[2],
                y: edge[3],
            },
            edge[4],
        )
    });

    compute_steal(edges, ras, xy_min);

    Ok(())
}

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "_bindings")]
fn grid_rasterize(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fill_accumulation_fold, m)?)?;
    m.add_function(wrap_pyfunction!(fill_accumulation_fold2, m)?)?;
    m.add_function(wrap_pyfunction!(fill_accumulation_async, m)?)?;
    Ok(())
}

fn fill(
    raster: &mut ArrayViewMut2<f64>,
    edges: impl Iterator<Item = Edge> + Send,
    xy_min: &Point<isize>,
) {
    let sh = raster.raw_dim();

    let solutions = edges
        .par_bridge()
        .fold(
            || {
                println!("running id on {:?}", current_thread_index());
                Array2::zeros(sh)
            },
            |mut im, edge| {
                fill_accumulation(&mut im, &edge.try_into().unwrap(), &xy_min);
                im
            },
        )
        .reduce_with(
            // || {
            //     println!("id for reduce on {:?}", current_thread_index());
            //     Array2::zeros(sh)
            // },
            |im0, im1| im0 + im1,
        )
        .unwrap();

    // for solution in &solutions {
    *raster += &solutions;
    // }
}
