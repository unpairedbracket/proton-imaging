use std::{
    fmt::Debug,
    ops::{Index, Mul, Sub},
};

use conv::{ApproxInto, NoError, UnwrapOk};

#[derive(Clone, Copy, Debug)]
pub struct Point<T: Copy + std::fmt::Debug> {
    pub x: T,
    pub y: T,
}

#[derive(Debug)]
pub struct Triangle<T: Copy + Debug>(pub Point<T>, pub Point<T>, pub Point<T>);

impl<T: Copy + Debug> Point<T> {
    pub fn as_approx<S: Copy + Debug>(&self) -> Point<S>
    where
        T: ApproxInto<S, Err = NoError>,
    {
        Point {
            x: self.x.approx_into().unwrap_ok(),
            y: self.y.approx_into().unwrap_ok(),
        }
    }
}

impl<T: Copy + Debug> Triangle<T> {
    pub fn as_approx<S: Copy + Debug>(&self) -> Triangle<S>
    where
        T: ApproxInto<S, Err = NoError>,
    {
        Triangle(self.0.as_approx(), self.1.as_approx(), self.2.as_approx())
    }
}

impl<T: Copy + Debug + Sub<T, Output = T> + Mul<T, Output = T>> Triangle<T> {
    pub fn area(&self) -> T {
        (self.1.x - self.0.x) * (self.2.y - self.0.y)
            - (self.2.x - self.0.x) * (self.1.y - self.0.y)
    }
}

impl<I: Copy + Debug> Triangle<I> {
    pub fn index_into<O: Copy + Debug>(
        &self,
        x_vals: &impl Index<(I, I), Output = O>,
        y_vals: &impl Index<(I, I), Output = O>,
    ) -> Triangle<O> {
        Triangle(
            Point {
                x: x_vals[(self.0.x, self.0.y)],
                y: y_vals[(self.0.x, self.0.y)],
            },
            Point {
                x: x_vals[(self.1.x, self.1.y)],
                y: y_vals[(self.1.x, self.1.y)],
            },
            Point {
                x: x_vals[(self.2.x, self.2.y)],
                y: y_vals[(self.2.x, self.2.y)],
            },
        )
    }
}
