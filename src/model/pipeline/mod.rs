//! Defines the `Pipeline` trait and its implementations

pub mod token;
pub mod span;
pub mod relation;
pub mod context;
pub mod composable;


use ort::session::{SessionInputs, SessionOutputs};
use crate::util::result::Result;
use crate::util::compose::Composable;
use super::{input::text::TextInput, output::decoded::SpanOutput, params::Parameters};


/// Defines a generic pre-processor
pub trait PreProcessor<'a, I, C>: Composable<I, (SessionInputs<'a, 'a>, C)> { }
impl<'a, I, C, T: Composable<I, (SessionInputs<'a, 'a>, C)>> PreProcessor<'a, I, C> for T { }


/// Defines a generic post-processor
pub trait PostProcessor<'a, O, C>: Composable<(SessionOutputs<'a, 'a>, C), O> { }
impl<'a, O, C, T: Composable<(SessionOutputs<'a, 'a>, C), O>> PostProcessor<'a, O, C> for T { }


/// Defines a generic pipeline
pub trait Pipeline<'a> {
    type Input;
    type Output;
    type Context;
    
    fn pre_processor(&self, params: &Parameters) -> impl PreProcessor<'a, Self::Input, Self::Context>;
    fn post_processor(&self, params: &Parameters) -> impl PostProcessor<'a, Self::Output, Self::Context>;

    fn to_composable(self, model: &'a super::Model, params: &'a Parameters) -> impl Composable<Self::Input, Self::Output> where Self: Sized {
        composable::ComposablePipeline::new(self, model, params)
    }
}
