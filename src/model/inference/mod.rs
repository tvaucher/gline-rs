//! The inferencing part, leveraging the `ort` ONNX wrapper

use std::path::Path;
use ort::session::{SessionInputs, SessionOutputs};
use ort::session::{builder::GraphOptimizationLevel, Session};
use crate::util::result::Result;
use super::params::Parameters;

/// Non-transparent wrapper around an `ort` session
pub struct Model {    
    session: Session,
}


impl Model {
    pub fn new<P: AsRef<Path>>(model_path: P, params: &Parameters) -> Result<Self> {        
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(params.threads)?
            .commit_from_file(model_path)?;

        Ok(Self {
            session,
        })
    }


    pub fn inference(&self, input: SessionInputs<'_, '_>) -> Result<SessionOutputs<'_, '_>> {
        Ok(self.session.run(input)?)
    }

}
