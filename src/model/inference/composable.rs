//! Tools to make a `Model` available as a `Composable` (see also `Model::to_composable()`)

use ::composable::Composable;
use crate::util::result::Result;
use super::super::{params::Parameters, pipeline::Pipeline};
use super::Model;

/// References a model, a pipeline and some parameters to implement `Composable`
pub struct ComposableModel<'a, P: Pipeline<'a>> {
    model: &'a Model,
    pipeline: &'a P,
    params: &'a Parameters,
}


impl<'a, P: Pipeline<'a>> ComposableModel<'a, P> {
    pub fn new(model: &'a Model, pipeline: &'a P, params: &'a Parameters) -> Self {
        Self { 
            model, 
            pipeline, 
            params 
        }
    }
}


impl<'a, P: Pipeline<'a>> Composable<P::Input, P::Output> for ComposableModel<'a, P> {
    fn apply(&self, input: P::Input) -> Result<P::Output> {
        self.model.inference(input, self.pipeline, self.params)
    }
}