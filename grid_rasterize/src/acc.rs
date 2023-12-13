use geometry::Point;
use ndarray::{s, Array, Array2};

#[derive(Debug)]
pub struct Edge {
    p0: Point<f64>,
    p1: Point<f64>,
    d_intensity: f64,
}

impl Edge {
    pub fn new(p0: Point<f64>, p1: Point<f64>, d_intensity: f64) -> Edge {
        if p0.y > p1.y {
            Edge {
                p1: p0,
                p0: p1,
                d_intensity: -d_intensity,
            }
        } else {
            Edge {
                p0,
                p1,
                d_intensity,
            }
        }
    }
}

/// Fill an accumumlation buffer
pub fn fill_accumulation(raster: &mut Array2<f64>, edge: &Edge, xy_min: &Point<isize>) {
    // Firstly flip the edge if it's 'downwards': exchange p0 <-> p1 and negate its weight
    let Edge {
        p0: Point { x: x0, y: y0 },
        p1: Point { x: x1, y: y1 },
        d_intensity,
    } = *edge;

    let islope = (x1 - x0) / (y1 - y0);

    let mut all_y = Array::range(y0.floor(), y1.ceil() + 1.0, 1.0);

    let len = all_y.len();

    all_y[0] = y0;
    all_y[len - 1] = y1;

    let points = Vec::from_iter(all_y.iter().map(|&y| Point {
        x: x0 + (y - y0) * islope,
        y,
    }));

    for edge_pair in points.windows(2) {
        let [edge_prev, edge_next] = edge_pair else {panic!()};
        let mut line = raster.slice_mut(s![.., (edge_prev.y.floor() as isize - xy_min.y) as usize]);
        let x_mns = if islope >= 0.0 {
            edge_prev.x
        } else {
            edge_next.x
        };
        let y_prev = edge_prev.y;
        let y_next = edge_next.y;

        let h = y_next - y_prev;
        let x_pls = x_mns + islope.abs() * h;
        let x_px_min = x_mns.floor() as isize;
        let x_px_max = 1 + x_pls.floor() as isize;

        let ix_min = (x_px_min - xy_min.x) as usize;
        let ix_max = (x_px_max - xy_min.x) as usize;

        if (x_px_max - x_px_min) == 1 {
            let x_avg = 0.5 * (x_mns + x_pls) - x_px_min as f64;
            line[ix_min] += (1.0 - x_avg) * h * d_intensity;
            line[ix_max] += x_avg * h * d_intensity;
        } else {
            let slope = islope.abs().recip();
            let dx0 = x_px_min as f64 - x_mns + 1.0;
            let dx1 = x_pls - x_px_max as f64 + 1.0;
            line[ix_min] += 0.5 * dx0 * dx0 * slope * d_intensity;
            line[ix_max] += 0.5 * dx1 * dx1 * slope * d_intensity;
            let mut centres = line.slice_mut(s![ix_min + 1..=ix_max - 1]);
            centres += slope * d_intensity;
            line[ix_min + 1] -= 0.5 * (1.0 - dx0) * (1.0 - dx0) * slope * d_intensity;
            line[ix_max - 1] -= 0.5 * (1.0 - dx1) * (1.0 - dx1) * slope * d_intensity;
        }
    }
}

/// Fill an accumumlation buffer
pub fn fill_accumulation_edges(raster: &mut Array2<f64>, edge: &Edge, xy_min: &Point<isize>) {
    // Firstly flip the edge if it's 'downwards': exchange p0 <-> p1 and negate its weight
    let Edge {
        p0: Point { x: x0, y: y0 },
        p1: Point { x: x1, y: y1 },
        d_intensity: _,
    } = *edge;

    let islope = (x1 - x0) / (y1 - y0);

    let mut all_y = Array::range(y0.floor(), y1.ceil() + 1.0, 1.0);

    let len = all_y.len();

    all_y[0] = y0;
    all_y[len - 1] = y1;

    let points = Vec::from_iter(all_y.iter().map(|&y| Point {
        x: x0 + (y - y0) * islope,
        y,
    }));

    for edge_pair in points.windows(2) {
        let [edge_prev, edge_next] = edge_pair else {panic!()};
        let mut line = raster.slice_mut(s![.., (edge_prev.y.floor() as isize - xy_min.y) as usize]);
        let x_mns = if islope >= 0.0 {
            edge_prev.x
        } else {
            edge_next.x
        };
        let y_prev = edge_prev.y;
        let y_next = edge_next.y;

        let h = y_next - y_prev;
        let x_pls = x_mns + islope.abs() * h;
        let x_px_min = x_mns.floor() as isize;
        let x_px_max = 1 + x_pls.floor() as isize;

        let ix_min = (x_px_min - xy_min.x) as usize;
        let ix_max = (x_px_max - xy_min.x) as usize;

        if (x_px_max - x_px_min) == 1 {
            let x_avg = 0.5 * (x_mns + x_pls) - x_px_min as f64;
            line[ix_min] += (1.0 - x_avg) * h;
            line[ix_max] += x_avg * h;
        } else {
            let slope = islope.abs().recip();
            let dx0 = x_px_min as f64 - x_mns + 1.0;
            let dx1 = x_pls - x_px_max as f64 + 1.0;
            line[ix_min] += 0.5 * dx0 * dx0 * slope;
            line[ix_max] += 0.5 * dx1 * dx1 * slope;
            let mut centres = line.slice_mut(s![ix_min + 1..=ix_max - 1]);
            centres += slope;
            line[ix_min + 1] -= 0.5 * (1.0 - dx0) * (1.0 - dx0) * slope;
            line[ix_max - 1] -= 0.5 * (1.0 - dx1) * (1.0 - dx1) * slope;
        }
    }
}
