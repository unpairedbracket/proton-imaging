use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use ndarray::{Array2, ArrayBase, Axis, Data, Ix2, OwnedRepr, ViewRepr};
use rayon::{
    current_num_threads,
    iter::plumbing::{bridge_unindexed, UnindexedProducer},
    prelude::{IntoParallelIterator, ParallelIterator},
};

pub struct Grid<D>
where
    D: Data<Elem = RootNode>,
{
    root_nodes: ArrayBase<D, Ix2>,
}

impl Grid<OwnedRepr<RootNode>> {
    pub fn new(pixel_shape: [usize; 2]) -> Self {
        let root_nodes = Array2::from_shape_fn(pixel_shape, |indices| RootNode::new(indices));
        Grid { root_nodes }
    }

    pub fn view(&self) -> Grid<ViewRepr<&'_ RootNode>> {
        Grid {
            root_nodes: self.root_nodes.view(),
        }
    }
}
impl<D> Grid<D>
where
    D: Data<Elem = RootNode>,
{
    pub fn traverse(&self, mut cb: impl FnMut(&Node, &str)) {
        for root in &self.root_nodes {
            root.start_traversal(&mut cb);
        }
    }
}

impl Grid<ViewRepr<&'_ RootNode>> {
    pub fn split_longest(self) -> (Self, Option<Self>) {
        let shape = self.root_nodes.shape();
        let max_axis = if shape[0] >= shape[1] { 0 } else { 1 };
        self.split(Axis(max_axis))
    }
    pub fn split(self, axis: Axis) -> (Self, Option<Self>) {
        let axlen = self.root_nodes.shape()[axis.0];
        if axlen == 1 {
            (self, None)
        } else {
            let (left, right) = self.root_nodes.split_at(axis, axlen / 2);
            (Grid { root_nodes: left }, Some(Grid { root_nodes: right }))
        }
    }
}

pub struct RootNode {
    indices: (usize, usize),
    inner: Node,
}

impl RootNode {
    fn new(indices: (usize, usize)) -> Self {
        let inner = Node::default();
        RootNode { indices, inner }
    }

    fn start_traversal(&self, cb: &mut impl FnMut(&Node, &str)) {
        self.inner.enter(format!("{:?}", self.indices), cb)
    }
}

#[derive(Default)]
pub struct Node {
    children: [Option<Box<Node>>; 4],
}

impl Node {
    fn enter(&self, id: String, cb: &mut impl FnMut(&Node, &str)) {
        cb(self, &id);
        for (child_index, child) in self.children.iter().enumerate() {
            if let Some(child_node) = child {
                child_node.enter(format!("{id}.{child_index}"), cb);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        thread,
        time::{Duration, Instant},
    };

    use rayon::{
        current_thread_index,
        prelude::{IntoParallelIterator, ParallelIterator},
    };

    use crate::qt::Grid;

    #[test]
    fn traverse_simple_grid() {
        let grid = Grid::new([2000, 2000]);
        // grid.traverse(|node, id| println!("In node {id}"));
        // let (grid_1, Some(grid_2)) = grid.view().split_longest() else {panic!()};
        // let (grid_1_1, Some(grid_1_2)) = grid_1.split_longest() else {panic!()};

        // grid_1_1.traverse(|node, id| println!("{id} in first quarter!"));
        // grid_2.traverse(|node, id| println!("{id} in second half!"));

        // grid.traverse(|node, id| println!("In node {id}"));
        // grid_1_2.traverse(|node, id| println!("{id} in second quarter!"));

        println!("Doing parallel iter");
        let t0 = Instant::now();
        let total_length = grid
            .into_par_iter()
            .fold(
                || {
                    println!("init on {:?}", current_thread_index());
                    thread::sleep(Duration::from_millis(10));
                    0
                },
                |total, id| total + id.len(),
            )
            // .map(|total| {
            //     println!("length {total} on {:?}", current_thread_index());
            //     total
            // })
            .reduce(|| 0, |acc, new| acc + new);
        println!("time taken: {:?}", t0.elapsed());
        println!("total length: {total_length}");
        let t0 = Instant::now();
        let things: Vec<_> = grid.into_par_iter().collect();
        println!("time taken: {:?}", t0.elapsed());
        thread::sleep(Duration::from_secs(30));
        let t0 = Instant::now();
        let total_length = things.into_iter().fold(0, |acc, new| acc + new.len());
        println!("time taken: {:?}", t0.elapsed());
        println!("total length: {total_length}");
    }
}

impl<'a> IntoParallelIterator for &'a Grid<OwnedRepr<RootNode>> {
    type Iter = Grid<ViewRepr<&'a RootNode>>;

    type Item = String;

    fn into_par_iter(self) -> Self::Iter {
        self.view()
    }
}

impl<'a> ParallelIterator for Grid<ViewRepr<&'a RootNode>> {
    type Item = String;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        let num_threads = current_num_threads();

        let prod = GridProducer {
            grid: self,
            split_count: Arc::new(num_threads.into()),
        };

        bridge_unindexed(prod, consumer)
        // bridge_unindexed(self.view(), consumer)
    }
}

struct GridProducer<'a> {
    grid: Grid<ViewRepr<&'a RootNode>>,
    split_count: Arc<AtomicUsize>,
}

impl UnindexedProducer for GridProducer<'_> {
    type Item = String;

    fn split(self) -> (Self, Option<Self>) {
        let mut count = self.split_count.load(Ordering::SeqCst);

        loop {
            // Check if the iterator is exhausted
            if let Some(new_count) = count.checked_sub(1) {
                match self.split_count.compare_exchange_weak(
                    count,
                    new_count,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => {
                        let (grid_left, grid_right) = self.grid.split_longest();
                        let right = grid_right.map(|grid| GridProducer {
                            grid,
                            split_count: self.split_count.clone(),
                        });
                        let left = GridProducer {
                            grid: grid_left,
                            split_count: self.split_count,
                        };
                        return (left, right);
                    }
                    Err(last_count) => count = last_count,
                }
            } else {
                return (self, None);
            }
        }
    }

    fn fold_with<F>(self, folder: F) -> F
    where
        F: rayon::iter::plumbing::Folder<Self::Item>,
    {
        let mut fol = Some(folder);
        self.grid.traverse(|_node, id| {
            let f = fol.take().unwrap().consume(id.to_string());
            fol = Some(f);
        });
        fol.take().unwrap()
    }
}

impl UnindexedProducer for Grid<ViewRepr<&'_ RootNode>> {
    type Item = String;

    fn split(self) -> (Self, Option<Self>) {
        self.split(Axis(1))
    }

    fn fold_with<F>(self, folder: F) -> F
    where
        F: rayon::iter::plumbing::Folder<Self::Item>,
    {
        let mut fol = Some(folder);
        self.traverse(|_node, id| {
            let f = fol.take().unwrap().consume(id.to_string());
            fol = Some(f);
        });
        fol.take().unwrap()
    }
}
