mod acc;

pub use acc::{fill_accumulation, fill_accumulation_edges, Edge};

use geometry::Point;
use ndarray::{Array2, ArrayViewMut2};
use rayon::{current_thread_index, iter::IntoParallelIterator, iter::ParallelIterator};

pub fn draw_edges_parallel<I>(
    raster: &mut ArrayViewMut2<f64>,
    edges: impl IntoParallelIterator<Item = Edge>,
    xy_min: &Point<isize>,
) {
    let sh = raster.raw_dim();

    let solution = edges
        .into_par_iter()
        .fold(
            || {
                println!("new split on {:?}", current_thread_index());
                Array2::zeros(sh)
            },
            |mut im, edge| {
                fill_accumulation(&mut im, &edge, xy_min);
                im
            },
        )
        .reduce(|| Array2::zeros(sh), |im0, im1| im0 + im1);

    (*raster) += &solution;
}

pub fn draw_edges_serial(
    raster: &mut Array2<f64>,
    edges: impl IntoIterator<Item = Edge>,
    xy_min: &Point<isize>,
) {
    for edge in edges {
        acc::fill_accumulation(raster, &edge, xy_min);
    }
}
