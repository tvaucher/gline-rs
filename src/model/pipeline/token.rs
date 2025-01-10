//! Pre-defined pipeline for token mode

use std::path::Path;
use crate::util::result::Result;
use super::super::super::text::{splitter::Splitter, tokenizer::Tokenizer};
use super::super::super::util::compose::{compose, composed};
use super::super::params::Parameters;
use super::super::{input, output};
use super::*;


/// Generic token-level pipeline
pub struct TokenPipeline<S, T> {
    splitter: S,
    tokenizer: T,
}

impl<'a, S: Splitter, T:Tokenizer> Pipeline<'a> for TokenPipeline<S, T> {
    fn pre_processor(&self, params: &Parameters) -> impl PreProcessor<'a> {
        composed![
            input::tokenized::RawToTokenized::new(&self.splitter, params.max_length),
            input::prompt::TokenizedToPrompt::default(),
            input::encoded::PromptsToEncoded::new(&self.tokenizer),
            input::tensors::token::EncodedToTensors::default(),
            input::tensors::token::TensorsToSessionInput::default()
        ]
    }

    fn post_processor(&self, params: &Parameters) -> impl PostProcessor<'a> {
        composed![
            output::tensors::SessionOutputToTensors::default(),            
            output::decoded::token::TensorsToDecoded::new(params.threshold),
            output::decoded::greedy::GreedySearch::new(params.flat_ner, params.multi_label)
        ]
    }
}


/// Specific implementation using HF tokenizer and default splitter
impl TokenPipeline<crate::text::splitter::RegexSplitter, crate::text::tokenizer::HFTokenizer> {
    pub fn new<P: AsRef<Path>>(tokenizer_path: P) -> Result<Self> {
        Ok(Self {
            splitter: crate::text::splitter::RegexSplitter::default(),
            tokenizer: crate::text::tokenizer::HFTokenizer::from_file(tokenizer_path)?,
        })
    }
}

/// Shorthand for the default token pipeline type (eases disambiguation when calling `GLiNER::new`)
pub type TokenMode = TokenPipeline<crate::text::splitter::RegexSplitter, crate::text::tokenizer::HFTokenizer>;


/// Specific GLiNER implementation using the default token-mode pipeline
impl super::super::GLiNER<TokenMode> {
    pub fn new<P: AsRef<Path>>(params: Parameters, tokenizer_path: P, model_path: P) -> Result<Self> {
        Ok(Self {            
            model: super::super::Model::new(model_path, &params)?,
            pipeline: TokenPipeline::new(tokenizer_path)?,
            params,
        })
    }
}