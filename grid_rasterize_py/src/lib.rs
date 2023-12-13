use geometry::Point;
use grid_rasterize::{fill_accumulation, Edge};
use ndarray::{
    parallel::prelude::{IntoParallelIterator, ParallelIterator},
    Array2, Axis,
};
use numpy::{PyReadonlyArray2, PyReadwriteArray2};
use pyo3::prelude::*;

#[pyfunction]
fn fill_accumulation_rs(
    mut raster: PyReadwriteArray2<f64>,
    edges: PyReadonlyArray2<f64>,
    xy_min: (isize, isize),
) -> PyResult<()> {
    let mut ras = raster.as_array_mut();
    let sh = ras.raw_dim();

    let xy_min = Point {
        x: xy_min.0,
        y: xy_min.1,
    };

    let edges = edges.as_array();

    let solution = edges
        .axis_iter(Axis(0))
        .into_par_iter()
        .fold(
            || Array2::zeros(sh),
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
        .reduce(|| Array2::zeros(sh), |im0, im1| im0 + im1);

    ras += &solution;
    Ok(())
}

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "_bindings")]
fn grid_rasterize_mod(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fill_accumulation_rs, m)?)?;
    Ok(())
}
