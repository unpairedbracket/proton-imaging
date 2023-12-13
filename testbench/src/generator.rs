struct Yielder<T> {
    current_value: Rc<Mutex<Option<T>>>,
}

impl<T> Yielder<T> {
    async fn yld(&self, value: T) -> () {
        {
            let mut opt = self.current_value.try_lock().unwrap();
            assert!(mem::replace(&mut *opt, Some(value)).is_none());
        }
        pender().await;
    }
}

struct Pender(bool);

fn pender() -> Pender {
    Pender(false)
}

impl Future for Pender {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.0 {
            Poll::Ready(())
        } else {
            self.0 = true;
            Poll::Pending
        }
    }
}

struct YieldConsumer<T> {
    future: Pin<Box<dyn Future<Output = ()>>>,
    waker: Waker,
    value: Rc<Mutex<Option<T>>>,
}

impl<T> YieldConsumer<T> {
    fn from_fn<F, A>(async_fn: impl Fn(Yielder<T>, A) -> F, args: A) -> Self
    where
        F: Future<Output = ()> + 'static,
    {
        let y = Yielder {
            current_value: Rc::new(Mutex::new(None)),
        };
        let value = y.current_value.clone();
        let waker = noop_waker();
        let future = async_fn(y, args).boxed_local();
        YieldConsumer {
            future,
            waker,
            value,
        }
    }
}

impl<T> Iterator for YieldConsumer<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let mut context = Context::from_waker(&self.waker);
        let mut pinned = Box::pin(&mut self.future);
        match pinned.as_mut().poll(&mut context) {
            Poll::Ready(()) => {
                return None;
            }
            Poll::Pending => {
                let mut v = self.value.try_lock().unwrap();
                match v.take() {
                    Some(t) => return Some(t),
                    None => panic!("yielded without a value"),
                };
            }
        }
    }
}

async fn yield_numbers(yielder: Yielder<i32>, _: ()) {
    for i in 0..10 {
        yielder.yld(i).await;
    }
}

#[test]
fn thing() {
    let g = YieldConsumer::from_fn(yield_numbers, ());

    for v in g {
        println!("{:?}", v);
    }
}
