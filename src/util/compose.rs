//! Simple system to allow for easy composition of:
//! * "functional traits" (as defined by `Composable` and its `apply` method)
//! * and/or functions 
//! * and/or closures.
//! 
//! Two composables can be composed through different means:
//! * the `Composable::compose(self, other)` method
//! * the `compose(one, other)` function
//! * the `composed![...]` macro
//! 
//! Basic example:
//! 
//! ```
//! use gliner::util::result::Result;
//! use gliner::util::compose::*;
//! 
//! struct AddTo { addend: usize }
//! 
//! impl Composable<usize,usize> for AddTo {
//!     fn apply(&self, input: usize) -> Result<usize> { Ok(input + self.addend) }
//! }
//! 
//! fn demo() -> Result<()> {
//!     let increment = AddTo { addend: 1 };
//!     let square = |x: usize| Ok(x * x);
//!     let composition = compose(increment, square);
//!     let result = composition.apply(2)?;
//!     assert_eq!(result, 9);
//!     Ok(())
//! }
//! ```

use std::marker::PhantomData;
use super::result::Result;


/// Defines a "functional" (composable) trait
pub trait Composable<I, O> {
    fn apply(&self, input: I) -> Result<O>;

    fn compose<T, P>(self, other: T) -> impl Composable<I, P> where 
        Self: Sized,
        T: Composable<O, P>,
    {
        compose(self, other)
    }
}

/// Makes any `Fn<I, O>` composable
impl<F, I, O> Composable<I, O> for F where 
    F: Fn(I) -> Result<O> 
{
    fn apply(&self, arg: I) -> Result<O> {
        self(arg)
    }
}

/// Composition
pub fn compose<T1, T2, I, O, X>(t1: T1, t2: T2) -> impl Composable<I, O> where 
    T1: Composable<I, X>,
    T2: Composable<X, O>,
{
    Composed::new(t1, t2)
}


/// Support for composition 
struct Composed<T1, T2, X> {
    t1: T1,
    t2: T2,
    _marker: PhantomData<X>,
}

impl<T1, T2, X> Composed<T1, T2, X> {
    pub fn new(t1: T1, t2: T2) -> Self {
        Self { t1, t2, _marker: PhantomData }
    }
}

impl<T1, T2, I, O, X> Composable<I, O> for Composed<T1, T2, X> where 
    T1: Composable<I, X>,
    T2: Composable<X, O>,
{
    fn apply(&self, input: I) -> Result<O> {
        self.t2.apply(self.t1.apply(input)?)
    }
}


/// Macro-rules for easy composition (owned)
#[macro_export]
macro_rules! composed {

    ($e:expr) => {{
        $e
    }};

    ($e:expr, $($es:expr),+ ) => {{
        let t1 = $crate::composed! { $e };
        let t2 = $crate::composed! { $($es),+ };
        $crate::util::compose::compose(t1, t2)
    }};

}

pub(crate) use composed;
