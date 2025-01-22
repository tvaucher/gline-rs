//! Tools to make a `Pipeline` available as a `Composable` (see also `Pipeline::to_composable()`)

use crate::util::result::Result;
use crate::util::compose::Composable;
use super::super::super::model::inference::Model;
use super::super::params::Parameters;
use super::Pipeline;


/// Embeds a pipeline and a reference to a model and some parameters to implement `Composable`
pub struct ComposablePipeline<'a, P: Pipeline<'a>> {
    pipeline: P,
    model: &'a Model,    
    params: &'a Parameters,
}


impl<'a, P: Pipeline<'a>> ComposablePipeline<'a, P> {
    pub fn new(pipeline: P, model: &'a Model, params: &'a Parameters) -> Self {
        Self { 
            pipeline,
            model,             
            params 
        }
    }
}


impl<'a, P: Pipeline<'a>> Composable<P::Input, P::Output> for ComposablePipeline<'a, P> {
    fn apply(&self, input: P::Input) -> Result<P::Output> {
        self.model.inference(input, &self.pipeline, self.params)
    }
}