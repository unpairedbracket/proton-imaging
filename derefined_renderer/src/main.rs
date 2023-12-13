use std::{fs::File, io, path::Path};

use amr_quadtree::QuadTree;
use grid_rasterize::Edge;
use ndarray::{Array2, ArrayView2, Axis, Ix2, OwnedRepr, Slice};
use ndarray_npy::{write_npy, NpzReader, ReadNpzError, WriteNpyError};
use ndarray_stats::QuantileExt;
use thiserror::Error;

fn main() {
    main2().unwrap();
}

fn main2() -> Result<(), FileError> {
    let mut data = NpzReader::new(File::open("./cover.npz").unwrap()).unwrap();
    println!("{:?}", data.names().unwrap());
    let mut x: Array2<f64> = data.by_name("X.npy").unwrap();
    x.slice_axis_inplace(Axis(0), Slice::new(26, Some(-26), 1));
    x.slice_axis_inplace(Axis(1), Slice::new(77, Some(-77), 1));
    let mut y: Array2<f64> = data.by_name("Y.npy").unwrap();
    y.slice_axis_inplace(Axis(0), Slice::new(26, Some(-26), 1));
    y.slice_axis_inplace(Axis(1), Slice::new(77, Some(-77), 1));

    let mut noise = NpzReader::new(File::open("./noise_xy.npz").unwrap()).unwrap();
    println!("{:?}", noise.names().unwrap());
    let mut x_n: Array2<f64> = noise.by_name("X.npy").unwrap();
    x_n.slice_axis_inplace(Axis(0), Slice::new(26, Some(-26), 1));
    x_n.slice_axis_inplace(Axis(1), Slice::new(77, Some(-77), 1));
    let mut y_n: Array2<f64> = noise.by_name("Y.npy").unwrap();
    y_n.slice_axis_inplace(Axis(0), Slice::new(26, Some(-26), 1));
    y_n.slice_axis_inplace(Axis(1), Slice::new(77, Some(-77), 1));

    println!("{:?}", x.shape());

    println!("Making qtree");
    // let q = get_randomised(2, level - 1, 0.01);
    let q = get_from_file("./cover_refine.npz", 6, 6).unwrap();

    println!("rasterizing");
    let overall_scale = 10.0;
    let x_with_noise = overall_scale * (&x + 3.0 * &x_n);
    let y_with_noise = overall_scale * (&y + 3.0 * &y_n);
    let raster =
        main_dxdy(x_with_noise.view(), y_with_noise.view(), &q) * overall_scale * overall_scale;

    println!("saving");
    let result = write_npy("./output.npy", &raster).unwrap();
    Ok(result)
}

#[cfg(False)]
fn main_gaussian() {
    let dx = |x: f64, y: f64| x * (-(x * x + y * y) / (10.0 * 10.0)).exp();
    let dy = |x: f64, y: f64| y * (-(x * x + y * y) / (10.0 * 10.0)).exp();
    let x = Array::from_shape_fn((65, 65), |(i, j)| {
        i as f64 + dx(i as f64 - 32.0, j as f64 - 32.0)
    });
    let y = Array::from_shape_fn((65, 65), |(i, j)| {
        j as f64 + dy(i as f64 - 32.0, j as f64 - 32.0)
    });

    let q = QuadTree::demo();

    let raster = main_dxdy(&x, &y, &q);

    write_npy("./output.npy", &raster).unwrap();
}

fn main_dxdy(x: ArrayView2<f64>, y: ArrayView2<f64>, qt: &QuadTree) -> Array2<f64> {
    println!(
        "{}, {}, {}, {}",
        x.min_skipnan(),
        x.max_skipnan(),
        y.min_skipnan(),
        y.max_skipnan()
    );
    let shape = (
        (x.max_skipnan() - x.min_skipnan()) as usize + 20,
        (y.max_skipnan() - y.min_skipnan()) as usize + 20,
    );
    let mut raster = Array2::zeros(shape);
    let xy_0 = geometry::Point {
        x: (*x.min_skipnan() as isize) - 10,
        y: (*y.min_skipnan() as isize) - 10,
    };

    for tri in qt.get_triangles(Some(x.shape())) {
        // println!("{tri:?}");
        let transformed = tri.index_into(&x, &y);
        let area_0 = tri.as_approx::<f64>().area();
        let area_transformed = transformed.area();

        let edge = Edge::new(
            transformed.1.as_approx(),
            transformed.0.as_approx(),
            area_0 / area_transformed,
        );

        grid_rasterize::fill_accumulation(&mut raster, &edge, &xy_0);

        let edge = Edge::new(
            transformed.2.as_approx(),
            transformed.1.as_approx(),
            area_0 / area_transformed,
        );
        grid_rasterize::fill_accumulation(&mut raster, &edge, &xy_0);

        let edge = Edge::new(
            transformed.0.as_approx(),
            transformed.2.as_approx(),
            area_0 / area_transformed,
        );

        grid_rasterize::fill_accumulation(&mut raster, &edge, &xy_0);
    }

    raster
}

// fn get_randomised(base_level: u32, level: u32, p: f64) -> QuadTree {
//     let mut tree = QuadTree::new();
//     for i in 0..(1 << base_level) {
//         for j in 0..(1 << base_level) {
//             tree.add_node_by_position(base_level, i, j);
//         }
//     }
//     let mut rng = rand::thread_rng();
//     for i in 0..(1 << level) {
//         for j in 0..(1 << level) {
//             let r: f64 = rng.gen();
//             if r < p {
//                 tree.add_node_by_position(level, i, j)
//             }
//         }
//     }
//     tree
// }

#[derive(Error, Debug)]
enum FileError {
    #[error(transparent)]
    FileRead(#[from] io::Error),
    #[error(transparent)]
    Npz(#[from] ReadNpzError),
    #[error(transparent)]
    Npy(#[from] WriteNpyError),
}

fn get_from_file(
    npz_file_name: impl AsRef<Path>,
    data_depth: u32,
    max_refine: u32,
) -> Result<QuadTree, FileError> {
    let mut data = NpzReader::new(File::open(npz_file_name)?)?;
    let base_refines: Array2<bool> = data.by_name("level_0.npy")?;
    let shape = base_refines.raw_dim();
    let s = shape[0].max(shape[1]);
    // Log2, rounded up
    let base_level = (s - 1).ilog2() + 1;

    println!("Making quadtree of depth {}", base_level + data_depth);
    let mut tree = QuadTree::with_max_depth(base_level + data_depth);

    for i in 0..shape[0] {
        for j in 0..shape[1] {
            tree.add_node_by_position(base_level, i, j);
        }
    }

    for level in 0..max_refine {
        println!("level {level}");
        let Ok(level_refines) = data.by_name::<OwnedRepr<bool>, Ix2>(&format!("level_{}.npy", level)) else {break};
        for i in 0..level_refines.shape()[0] {
            for j in 0..level_refines.shape()[1] {
                if level_refines[(i, j)] {
                    tree.add_node_by_position(base_level + level + 1, 2 * i, 2 * j);
                    tree.add_node_by_position(base_level + level + 1, 2 * i, 2 * j + 1);
                    tree.add_node_by_position(base_level + level + 1, 2 * i + 1, 2 * j);
                    tree.add_node_by_position(base_level + level + 1, 2 * i + 1, 2 * j + 1);
                }
            }
        }
    }

    Ok(tree)
}

// trait IntoGrPoint<T: Copy + std::fmt::Debug> {
//     fn to(&self) -> grid_rasterize::Point<T>;
// }

// impl IntoGrPoint<f64> for amr_quadtree::Point<f64> {
//     fn to(&self) -> grid_rasterize::Point<f64> {
//         grid_rasterize::Point {
//             x: self.x,
//             y: self.y,
//         }
//     }
// }

#[test]
fn test_log2() {
    println!("{} {}", 5_usize.ilog2(), 4_usize.ilog2() + 1);
    println!("{} {}", 6_usize.ilog2(), 5_usize.ilog2() + 1);
    println!("{} {}", 7_usize.ilog2(), 6_usize.ilog2() + 1);
    println!("{} {}", 8_usize.ilog2(), 7_usize.ilog2() + 1);
    println!("{} {}", 9_usize.ilog2(), 8_usize.ilog2() + 1);
    println!("{} {}", 10_usize.ilog2(), 9_usize.ilog2() + 1);
}
