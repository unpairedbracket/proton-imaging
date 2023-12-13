use geometry::{Point, Triangle};

pub type GridPoint = Point<usize>;

#[derive(Debug)]
pub enum H {
    Left,
    Right,
}

#[derive(Debug)]
pub enum V {
    Top,
    Bottom,
}

#[derive(Debug)]
pub enum Direction {
    Vertical(V),
    Horizontal(H),
}

impl Direction {
    pub fn idx(&self) -> usize {
        match self {
            Direction::Horizontal(H::Right) => 0,
            Direction::Vertical(V::Top) => 1,
            Direction::Horizontal(H::Left) => 2,
            Direction::Vertical(V::Bottom) => 3,
        }
    }

    pub fn left() -> Direction {
        Direction::Horizontal(H::Left)
    }

    pub fn right() -> Direction {
        Direction::Horizontal(H::Right)
    }

    pub fn up() -> Direction {
        Direction::Vertical(V::Top)
    }

    pub fn down() -> Direction {
        Direction::Vertical(V::Bottom)
    }
}

#[derive(Debug)]
pub struct ChildPosition(V, H);

impl ChildPosition {
    pub fn idx(&self) -> usize {
        use {H::*, V::*};

        match self {
            ChildPosition(Bottom, Left) => 0,
            ChildPosition(Bottom, Right) => 1,
            ChildPosition(Top, Left) => 2,
            ChildPosition(Top, Right) => 3,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct NodeIndex(usize);

impl From<usize> for NodeIndex {
    fn from(value: usize) -> Self {
        NodeIndex(1 + value)
    }
}

impl From<Option<usize>> for NodeIndex {
    fn from(value: Option<usize>) -> Self {
        match value {
            None => NodeIndex(0),
            Some(int) => NodeIndex(1 + int),
        }
    }
}

impl From<NodeIndex> for Option<usize> {
    fn from(value: NodeIndex) -> Self {
        match value {
            NodeIndex(0) => None,
            NodeIndex(n) => Some(n - 1),
        }
    }
}

impl std::fmt::Debug for NodeIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Option::<usize>::from(*self).fmt(f)
    }
}

#[derive(Debug)]
pub struct Node {
    id: NodeIndex,
    _parent: NodeIndex,
    connected: [bool; 4],
    children: [NodeIndex; 4],
    completed: bool,
}

impl Node {
    fn root() -> Node {
        let mut r = Node::new(None.into(), None.into());
        r.completed = true;
        r
    }

    fn new(parent: NodeIndex, id: NodeIndex) -> Node {
        Node {
            id,
            _parent: parent,
            connected: [false; 4],
            children: [None.into(); 4],
            completed: false,
        }
    }

    fn add_child(&mut self, which: ChildPosition, id: NodeIndex) -> Result<Node, NodeIndex> {
        match Option::<usize>::from(self.children[which.idx()]) {
            Some(old_child_id) => Err(NodeIndex::from(Some(old_child_id)))?,
            None => {
                self.children[which.idx()] = id;
                Ok(Node::new(self.id, id))
            }
        }
    }

    fn iterate_preorder(
        &self,
        storage: &Vec<Node>,
        level: u32,
        i: usize,
        j: usize,
        closure: &mut impl FnMut(u32, usize, usize, &Node),
    ) {
        closure(level, i, j, &self);
        if let Some(child) =
            Option::<usize>::from(self.children[ChildPosition(V::Bottom, H::Left).idx()])
        {
            storage[child].iterate_preorder(storage, level + 1, i * 2, j * 2, closure);
        }
        if let Some(child) =
            Option::<usize>::from(self.children[ChildPosition(V::Bottom, H::Right).idx()])
        {
            storage[child].iterate_preorder(storage, level + 1, i * 2 + 1, j * 2, closure);
        }
        if let Some(child) =
            Option::<usize>::from(self.children[ChildPosition(V::Top, H::Left).idx()])
        {
            storage[child].iterate_preorder(storage, level + 1, i * 2, j * 2 + 1, closure);
        }
        if let Some(child) =
            Option::<usize>::from(self.children[ChildPosition(V::Top, H::Right).idx()])
        {
            storage[child].iterate_preorder(storage, level + 1, i * 2 + 1, j * 2 + 1, closure);
        }
    }

    fn push_triangles(&self, i: usize, j: usize, scale: usize, dest: &mut Vec<Triangle<usize>>) {
        let (x0, xc, x1) = (2 * i * scale, (2 * i + 1) * scale, 2 * (i + 1) * scale);
        let (y0, yc, y1) = (2 * j * scale, (2 * j + 1) * scale, 2 * (j + 1) * scale);

        if self.connected[Direction::Horizontal(H::Right).idx()] {
            if Option::<usize>::from(self.children[ChildPosition(V::Bottom, H::Right).idx()])
                .is_none()
            {
                dest.push(Triangle(
                    GridPoint { x: xc, y: yc },
                    GridPoint { x: x1, y: y0 },
                    GridPoint { x: x1, y: yc },
                ));
            }
            if Option::<usize>::from(self.children[ChildPosition(V::Top, H::Right).idx()]).is_none()
            {
                dest.push(Triangle(
                    GridPoint { x: xc, y: yc },
                    GridPoint { x: x1, y: yc },
                    GridPoint { x: x1, y: y1 },
                ));
            }
        } else {
            dest.push(Triangle(
                GridPoint { x: xc, y: yc },
                GridPoint { x: x1, y: y0 },
                GridPoint { x: x1, y: y1 },
            ));
        }

        if self.connected[Direction::Vertical(V::Top).idx()] {
            if Option::<usize>::from(self.children[ChildPosition(V::Top, H::Right).idx()]).is_none()
            {
                dest.push(Triangle(
                    GridPoint { x: xc, y: yc },
                    GridPoint { x: x1, y: y1 },
                    GridPoint { x: xc, y: y1 },
                ));
            }
            if Option::<usize>::from(self.children[ChildPosition(V::Top, H::Left).idx()]).is_none()
            {
                dest.push(Triangle(
                    GridPoint { x: xc, y: yc },
                    GridPoint { x: xc, y: y1 },
                    GridPoint { x: x0, y: y1 },
                ));
            }
        } else {
            dest.push(Triangle(
                GridPoint { x: xc, y: yc },
                GridPoint { x: x1, y: y1 },
                GridPoint { x: x0, y: y1 },
            ));
        }

        if self.connected[Direction::Horizontal(H::Left).idx()] {
            if Option::<usize>::from(self.children[ChildPosition(V::Top, H::Left).idx()]).is_none()
            {
                dest.push(Triangle(
                    GridPoint { x: xc, y: yc },
                    GridPoint { x: x0, y: y1 },
                    GridPoint { x: x0, y: yc },
                ));
            }
            if Option::<usize>::from(self.children[ChildPosition(V::Bottom, H::Left).idx()])
                .is_none()
            {
                dest.push(Triangle(
                    GridPoint { x: xc, y: yc },
                    GridPoint { x: x0, y: yc },
                    GridPoint { x: x0, y: y0 },
                ));
            }
        } else {
            dest.push(Triangle(
                GridPoint { x: xc, y: yc },
                GridPoint { x: x0, y: y1 },
                GridPoint { x: x0, y: y0 },
            ));
        }

        if self.connected[Direction::Vertical(V::Bottom).idx()] {
            if Option::<usize>::from(self.children[ChildPosition(V::Bottom, H::Left).idx()])
                .is_none()
            {
                dest.push(Triangle(
                    GridPoint { x: xc, y: yc },
                    GridPoint { x: x0, y: y0 },
                    GridPoint { x: xc, y: y0 },
                ));
            }
            if Option::<usize>::from(self.children[ChildPosition(V::Bottom, H::Right).idx()])
                .is_none()
            {
                dest.push(Triangle(
                    GridPoint { x: xc, y: yc },
                    GridPoint { x: xc, y: y0 },
                    GridPoint { x: x1, y: y0 },
                ));
            }
        } else {
            dest.push(Triangle(
                GridPoint { x: xc, y: yc },
                GridPoint { x: x0, y: y0 },
                GridPoint { x: x1, y: y0 },
            ));
        }
    }
}

#[derive(Debug)]
pub struct QuadTree {
    root: Node,
    nodes: Vec<Node>,
    max_depth: u32,
}

impl QuadTree {
    pub fn new() -> QuadTree {
        QuadTree::with_max_depth(0)
    }

    pub fn with_max_depth(depth: u32) -> QuadTree {
        let root = Node::root();
        QuadTree {
            root,
            nodes: Vec::new(),
            max_depth: depth,
        }
    }

    pub fn ind_to_path(depth: u32, i: usize, j: usize) -> Vec<ChildPosition> {
        if i.max(j) >= 2_usize.pow(depth) {
            panic!("that's too big")
        }

        let mut path = Vec::with_capacity(depth.try_into().unwrap());
        for shift in (0..depth).rev() {
            path.push(ChildPosition(
                if j >> shift & 0b1 == 0 {
                    V::Bottom
                } else {
                    V::Top
                },
                if i >> shift & 0b1 == 0 {
                    H::Left
                } else {
                    H::Right
                },
            ));
        }
        path
    }

    fn get_or_create_node(&mut self, path: Vec<ChildPosition>) -> NodeIndex {
        self.max_depth = self.max_depth.max(path.len() as u32);
        let mut node = &mut self.root;
        let mut next_new_id = self.nodes.len();
        for turn in path {
            let next_id = match Option::from(node.children[turn.idx()]) {
                None => {
                    // Make a new node along the path
                    let new_child = node.add_child(turn, next_new_id.into()).unwrap();
                    self.nodes.push(new_child);
                    next_new_id
                }
                Some(id) => id,
            };
            next_new_id = self.nodes.len();
            node = &mut self.nodes[next_id];
        }
        node.id
    }

    pub fn add_node_by_position(&mut self, level: u32, i: usize, j: usize) {
        self.ensure_node_connected(level, i, j, None)
    }

    pub fn refine_node(&mut self, level: u32, i: usize, j: usize) {
        self.add_node_by_position(level + 1, 2 * i, 2 * j);
        self.add_node_by_position(level + 1, 2 * i, 2 * j + 1);
        self.add_node_by_position(level + 1, 2 * i + 1, 2 * j);
        self.add_node_by_position(level + 1, 2 * i + 1, 2 * j + 1);
    }

    fn ensure_node_connected(
        &mut self,
        level: u32,
        i: usize,
        j: usize,
        direction: Option<Direction>,
    ) {
        let path = QuadTree::ind_to_path(level, i, j);
        // println!("connecting node at path {path:?} in direction {direction:?}");
        let node_id = Option::<usize>::from(self.get_or_create_node(path));
        let new_node = if let Some(new_id) = node_id {
            &mut self.nodes[new_id]
        } else {
            &mut self.root
        };

        if let Some(direction) = direction {
            new_node.connected[direction.idx()] = true;
        }

        if !new_node.completed {
            new_node.completed = true;
            if i > 0 {
                self.ensure_node_connected(level - 1, (i - 1) / 2, j / 2, Some(Direction::right()))
            }
            if (i + 1) < 2_usize.pow(level) {
                self.ensure_node_connected(level - 1, (i + 1) / 2, j / 2, Some(Direction::left()))
            }
            if j > 0 {
                self.ensure_node_connected(level - 1, i / 2, (j - 1) / 2, Some(Direction::up()))
            }
            if (j + 1) < 2_usize.pow(level) {
                self.ensure_node_connected(level - 1, i / 2, (j + 1) / 2, Some(Direction::down()))
            }
        }
    }

    pub fn for_each_node_with_coords(&self, closure: &mut impl FnMut(u32, usize, usize, &Node)) {
        self.root.iterate_preorder(&self.nodes, 0, 0, 0, closure)
    }

    pub fn get_levels(&self) -> Vec<u32> {
        let mut levels = vec![0; 2_usize.pow(self.max_depth * 2)];
        let arr_stride = 2_usize.pow(self.max_depth);
        self.for_each_node_with_coords(&mut |level, i, j, _node| {
            let stride = 2_usize.pow(self.max_depth - level);
            for ii in stride * i..stride * (i + 1) {
                for jj in stride * j..stride * (j + 1) {
                    let idx = jj * arr_stride + ii;
                    levels[idx] = level;
                }
            }
        });
        levels
    }

    pub fn get_triangles(&self, max_indices: Option<&[usize]>) -> Vec<Triangle<usize>> {
        let mut tris = Vec::new();
        self.for_each_node_with_coords(&mut |level, i, j, node| {
            let stride = 1 << (self.max_depth - level);
            if let Some(&[sx, sy]) = max_indices {
                if (2 * (i + 1) * stride > sx) | (2 * (j + 1) * stride > sy) {
                    return;
                }
            }
            node.push_triangles(i, j, stride, &mut tris)
        });
        tris
    }

    pub fn demo() -> QuadTree {
        let mut tree = QuadTree::new();
        for i in 0..4 {
            for j in 0..4 {
                tree.add_node_by_position(2, i, j);
            }
        }
        tree.add_node_by_position(5, 16, 16);
        tree.add_node_by_position(5, 16, 17);
        tree.add_node_by_position(5, 17, 16);
        tree.add_node_by_position(5, 17, 17);
        tree
    }
}

#[test]
fn print_test_tree_levels() {
    let tree = QuadTree::demo();

    let levels = tree.get_levels();
    let stride = 2_usize.pow(tree.max_depth);
    println!("[");
    for i in (0..stride).rev() {
        println!("{:?},", &levels[stride * i..stride * (1 + i)]);
    }
    println!("]");
}

#[test]
fn print_test_tree_nodes() {
    let mut tree = QuadTree::new();
    tree.add_node_by_position(1, 0, 0);

    println!("{:?}", tree)
}

#[test]
fn test_tris() {
    let tree = QuadTree::demo();

    let tris = tree.get_triangles(None);
    print!("[");
    for tri in tris {
        println!(
            "[[{}, {}], [{}, {}], [{}, {}]],",
            tri.0.x, tri.0.y, tri.1.x, tri.1.y, tri.2.x, tri.2.y
        );
    }
    println!("]");
}
