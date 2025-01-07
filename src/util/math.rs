use num_traits::Float;


pub fn sigmoid<T: Float>(x: T) -> T {
    T::one() / (T::one() + (-x).exp())
}