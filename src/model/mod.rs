//! Everything about pre-/post-processing and inferencing

pub mod params;
pub mod pipeline;
pub mod input;
pub mod output;
pub mod inference;

use super::util::compose::Composable;
use crate::util::result::Result;
use pipeline::Pipeline;
use params::Parameters;
use input::text::TextInput;
use output::decoded::SpanOutput;
use inference::Model;


/// Generic GLiNER, to be parametrized by a specific pipeline (see implementations within the pipeline module)
pub struct GLiNER<T> {
    params: Parameters,
    model: Model,
    pipeline: T,
}


impl<'a, T: Pipeline<'a>> GLiNER<T> {
    pub fn inference(&'a self, input: TextInput) -> Result<SpanOutput> {        
        // pre-process
        let (input, meta) = self.pipeline.pre_processor(&self.params).apply(input)?;
        // inference
        let output = self.model.inference(input)?;                
        // post-process
        let output = self.pipeline.post_processor(&self.params).apply((output, meta))?;        
        // ok
        Ok(output)
    }
}
