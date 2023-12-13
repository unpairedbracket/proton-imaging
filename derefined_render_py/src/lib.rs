use pyo3::prelude::*;

mod quadtree;

/// A Python module implemented in Rust.
#[pymodule]
fn derefined_render_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<quadtree::PyQuadtree>()?;
    Ok(())
}
