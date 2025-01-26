//! The inferencing part, leveraging the `ort` ONNX wrapper
pub mod composable;

use std::path::Path;
use ort::session::{SessionInputs, SessionOutputs};
use ort::session::{builder::GraphOptimizationLevel, Session};
use crate::util::compose::Composable;
use crate::util::result::Result;
use super::params::{Parameters, RuntimeParameters};
use super::pipeline::Pipeline;


/// A `Model` can load an ONNX model, and run it using the provided pipeline.
pub struct Model {    
    session: Session,
}


impl Model {    
    pub fn new<P: AsRef<Path>>(model_path: P, params: RuntimeParameters) -> Result<Self> {
        let session = Session::builder()?
            .with_execution_providers(params.execution_providers)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(params.threads)?
            .commit_from_file(model_path)?;

        Ok(Self {
            session,
        })
    }

    pub fn inference<'a, P: Pipeline<'a>>(&'a self, input: P::Input, pipeline: &P, params: &Parameters) -> Result<P::Output> {
        // pre-process
        let (input, context) = pipeline.pre_processor(params).apply(input)?;
        // inference
        let output = self.run(input)?;                
        // post-process
        let output = pipeline.post_processor(params).apply((output, context))?;        
        // ok
        Ok(output)
    }

    pub fn to_composable<'a, P: Pipeline<'a>>(&'a self, pipeline: &'a P, params: &'a Parameters) -> impl Composable<P::Input, P::Output> {
        composable::ComposableModel::new(self, pipeline, params)
    }


    fn run(&self, input: SessionInputs<'_, '_>) -> Result<SessionOutputs<'_, '_>> {
        Ok(self.session.run(input)?)
    }


}
