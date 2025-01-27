//! Pre-defined pipeline for span mode

use std::path::Path;
use ::composable::*;
use crate::util::result::Result;
use super::super::super::text::{splitter::Splitter, tokenizer::Tokenizer};
use super::super::params::{Parameters, RuntimeParameters};
use super::super::{input, output};
use super::*;
use context::EntityContext;


/// Generic span-level pipeline
pub struct SpanPipeline<S, T> {
    splitter: S,
    tokenizer: T,
}

impl<'a, S: Splitter, T:Tokenizer> Pipeline<'a> for SpanPipeline<S, T> {
    type Input = TextInput;
    type Output = SpanOutput;
    type Context = EntityContext;

    fn pre_processor(&self, params: &Parameters) -> impl PreProcessor<'a, Self::Input, Self::Context>{
        composed![
            input::tokenized::RawToTokenized::new(&self.splitter, params.max_length),
            input::prompt::TokenizedToPrompt::default(),
            input::encoded::PromptsToEncoded::new(&self.tokenizer),
            input::tensors::span::EncodedToTensors::new(params.max_width),
            input::tensors::span::TensorsToSessionInput::default()
        ]
    }

    fn post_processor(&self, params: &Parameters) -> impl PostProcessor<'a, Self::Output, Self::Context> {
        composed![
            output::tensors::SessionOutputToTensors::default(),
            output::decoded::span::TensorsToDecoded::new(params.threshold, params.max_width),            
            output::decoded::greedy::GreedySearch::new(params.flat_ner, params.multi_label)
        ]
    }
}

/// Specific implementation using HF tokenizer and default splitter
impl SpanPipeline<crate::text::splitter::RegexSplitter, crate::text::tokenizer::HFTokenizer> {
    pub fn new<P: AsRef<Path>>(tokenizer_path: P) -> Result<Self> {
        Ok(Self {
            splitter: crate::text::splitter::RegexSplitter::default(),
            tokenizer: crate::text::tokenizer::HFTokenizer::from_file(tokenizer_path)?,
        })
    }
}

/// Shorthand for the default span pipeline type (eases disambiguation when calling `GLiNER::new`)
pub type SpanMode = SpanPipeline<crate::text::splitter::RegexSplitter, crate::text::tokenizer::HFTokenizer>;


/// Specific GLiNER implementation using the default span-mode pipeline
impl super::super::GLiNER<SpanMode> {
    pub fn new<P: AsRef<Path>>(params: Parameters, runtime_params: RuntimeParameters, tokenizer_path: P, model_path: P) -> Result<Self> {
        Ok(Self {            
            model: super::super::Model::new(model_path, runtime_params)?,
            pipeline: SpanPipeline::new(tokenizer_path)?,
            params,
        })
    }
}