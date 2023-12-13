use geometry::Point;
use grid_rasterize::Edge;
use ndarray::{Array2, ArrayView2};
use ndarray_stats::QuantileExt;
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

use amr_quadtree::QuadTree;

#[allow(dead_code)]
#[derive(Debug)]
#[pyclass]
pub struct PyQuadtree {
    base_level: u32,
    base_shape: (usize, usize),
    inner: QuadTree,
}

#[pymethods]
impl PyQuadtree {
    #[new]
    fn new(base_shape: (usize, usize)) -> Self {
        let base_level_x = match (base_shape.0 - 1).checked_ilog2() {
            Some(n) => n + 1,
            None => 0,
        };
        let base_level_y = match (base_shape.1 - 1).checked_ilog2() {
            Some(n) => n + 1,
            None => 0,
        };
        let base_level = base_level_x.max(base_level_y);
        let mut inner = QuadTree::new();

        for i in 0..base_shape.0 {
            for j in 0..base_shape.1 {
                inner.add_node_by_position(base_level, i, j);
            }
        }

        PyQuadtree {
            base_level,
            base_shape,
            inner,
        }
    }

    fn pritn(&self) {
        println!("{self:?}")
    }

    fn add_refinement_levels(&mut self, levels: Vec<Option<PyReadonlyArray2<bool>>>) {
        for (level, refine) in levels.into_iter().enumerate() {
            let level = level as u32;
            println!("level {level}");
            if let Some(refine) = refine {
                let refine = refine.as_array();
                for i in 0..refine.shape()[0] {
                    for j in 0..refine.shape()[1] {
                        if refine[(i, j)] {
                            self.inner.refine_node(self.base_level + level, i, j)
                        }
                    }
                }
            }
        }
    }

    fn render_grid<'py>(
        &self,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray2<f64>,
        py: Python<'py>,
    ) -> Py<PyArray2<f64>> {
        let (x, y) = (x.as_array(), y.as_array());

        let raster = self.render_function(x, y, grid_rasterize::fill_accumulation);
        raster.to_pyarray(py).to_owned()
    }

    fn render_edges<'py>(
        &self,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray2<f64>,
        py: Python<'py>,
    ) -> Py<PyArray2<f64>> {
        let (x, y) = (x.as_array(), y.as_array());

        let raster = self.render_function(x, y, grid_rasterize::fill_accumulation_edges);
        raster.to_pyarray(py).to_owned()
    }
}

impl PyQuadtree {
    fn render_function(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView2<f64>,
        render_function: impl Fn(&mut Array2<f64>, &Edge, &Point<isize>),
    ) -> Array2<f64> {
        let (x_min, x_max, y_min, y_max) = (
            x.min_skipnan(),
            x.max_skipnan(),
            y.min_skipnan(),
            y.max_skipnan(),
        );
        println!("{}, {}, {}, {}", x_min, x_max, y_min, y_max);

        let shape = ((x_max - x_min) as usize + 20, (y_max - y_min) as usize + 20);

        let mut raster = Array2::zeros(shape);

        let xy_0 = Point {
            x: (*x_min as isize) - 10,
            y: (*y_min as isize) - 10,
        };

        for tri in self.inner.get_triangles(Some(x.shape())) {
            // println!("{tri:?}");
            let transformed = tri.index_into(&x, &y);
            let area_0 = tri.area() as f64;
            let area_transformed = transformed.area();

            let edge = Edge::new(
                transformed.1.as_approx(),
                transformed.0.as_approx(),
                area_0 / area_transformed,
            );

            render_function(&mut raster, &edge, &xy_0);

            let edge = Edge::new(
                transformed.2.as_approx(),
                transformed.1.as_approx(),
                area_0 / area_transformed,
            );
            render_function(&mut raster, &edge, &xy_0);

            let edge = Edge::new(
                transformed.0.as_approx(),
                transformed.2.as_approx(),
                area_0 / area_transformed,
            );

            render_function(&mut raster, &edge, &xy_0);
        }
        raster
    }
}
