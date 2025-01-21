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
pub struct RelationPipeline<S, T> {
    token_pipeline: TokenPipeline<S, T>,
    relation_schema: RelationSchema,
}


impl<'a, S: Splitter, T:Tokenizer> Pipeline<'a> for RelationPipeline<S, T> {
    type Input = SpanOutput;
    type Output = RelationOutput;

    fn pre_processor(&self, params: &Parameters) -> impl PreProcessor<'a, Self::Input> {
        composed![
            SpanOutputToRelationInput::new(&self.relation_schema),
            RelationInputToTextInput::default(),
            self.token_pipeline.pre_processor(params)            
        ]        
    }

    fn post_processor(&self, params: &Parameters) -> impl PostProcessor<'a, Self::Output> {
        composed![
            self.token_pipeline.post_processor(params),
            SpanOutputToRelationOutput::default()
        ]
    }
}

impl<S, T> RelationPipeline<S, T> {
    pub fn new(token_pipeline: TokenPipeline<S, T>, relation_schema: RelationSchema) -> Self {
        Self {
            token_pipeline,
            relation_schema,
        }
    }
}

/// Builds a default relation extraction pipeline
impl RelationPipeline<crate::text::splitter::RegexSplitter, crate::text::tokenizer::HFTokenizer> {
    pub fn default<P: AsRef<Path>>(tokenizer_path: P, relation_schema: RelationSchema) -> Result<Self> {
        Ok(RelationPipeline::new(TokenPipeline::new(tokenizer_path)?, relation_schema))
    }
}
