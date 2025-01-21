//! Pre-defined pipeline for Relation Extraction

use std::path::Path;
use super::token::TokenPipeline;
use crate::model::input::relation::schema::RelationSchema;
use crate::model::input::relation::{RelationInputToTextInput, SpanOutputToRelationInput};
use crate::model::output::relation::{RelationOutput, SpanOutputToRelationOutput};
use crate::text::{splitter::Splitter, tokenizer::Tokenizer};
use crate::util::compose::{compose, composed};
use super::super::params::Parameters;
use super::*;

/// Relation Extraction pipeline
/// 
/// Re-uses the token-level pipeline (see `TokenPipline`)
pub struct RelationPipeline<'a, S, T> {
    token_pipeline: TokenPipeline<S, T>,
    relation_schema: &'a RelationSchema,
}


impl<'a, S: Splitter, T:Tokenizer> Pipeline<'a> for RelationPipeline<'a, S, T> {
    type Input = SpanOutput;
    type Output = RelationOutput;
    type Context = TensorsMeta;

    fn pre_processor(&self, params: &Parameters) -> impl PreProcessor<'a, Self::Input, Self::Context> {
        composed![
            SpanOutputToRelationInput::new(&self.relation_schema),
            RelationInputToTextInput::default(),
            self.token_pipeline.pre_processor(params)            
        ]        
    }

    fn post_processor(&self, params: &Parameters) -> impl PostProcessor<'a, Self::Output, Self::Context> {
        composed![
            self.token_pipeline.post_processor(params),
            SpanOutputToRelationOutput::new(&self.relation_schema)
        ]
    }
}

impl<'a, S, T> RelationPipeline<'a, S, T> {
    pub fn new(token_pipeline: TokenPipeline<S, T>, relation_schema: &'a RelationSchema) -> Self {
        Self {
            token_pipeline,
            relation_schema,
        }
    }
}

/// Builds a default relation extraction pipeline
impl<'a> RelationPipeline<'a, crate::text::splitter::RegexSplitter, crate::text::tokenizer::HFTokenizer> {
    pub fn default<P: AsRef<Path>>(tokenizer_path: P, relation_schema: &'a RelationSchema) -> Result<Self> {
        Ok(RelationPipeline::new(TokenPipeline::new(tokenizer_path)?, relation_schema))
    }
}
