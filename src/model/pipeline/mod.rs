//! Defines the `Pipeline` trait and its implementations

pub mod token;
pub mod span;

use ort::session::{SessionInputs, SessionOutputs};
use crate::util::{error::IndexError, result::Result};
use crate::util::compose::Composable;
use crate::text::span::Span;
use crate::text::token::Token;
use super::{input::text::TextInput, output::decoded::SpanOutput, params::Parameters};


/// Defines a generic pre-processor
pub trait PreProcessor<'a>: Composable<TextInput, (SessionInputs<'a, 'a>, TensorsMeta)> { }
impl<'a, T: Composable<TextInput, (SessionInputs<'a, 'a>, TensorsMeta)>> PreProcessor<'a> for T { }


/// Defines a generic post-processor
pub trait PostProcessor<'a>: Composable<(SessionOutputs<'a, 'a>, TensorsMeta), SpanOutput> { }
impl<'a, T: Composable<(SessionOutputs<'a, 'a>, TensorsMeta), SpanOutput>> PostProcessor<'a> for T { }


/// Defines a generic pipeline
pub trait Pipeline<'a> {
    fn pre_processor(&self, params: &Parameters) -> impl PreProcessor<'a>;
    fn post_processor(&self, params: &Parameters) -> impl PostProcessor<'a>;
}



/// Data to be transmitted, beside the tensors themselves, from pre-processing to post-processing
pub struct TensorsMeta {
    pub texts: Vec<String>,
    pub tokens: Vec<Vec<Token>>,
    pub entities: Vec<String>,
    pub num_words: usize,
}

impl TensorsMeta {
    /// Creates a span given the necessary indexes and the tensor meta data.
    /// Currently panics if the offsets are not correct (might return an error in the future)
    pub fn create_span(&self, sequence_id: usize, start_token: usize, end_token: usize, class: usize, probability: f32) -> Result<Span> {
        let sequence = self.tokens.get(sequence_id).ok_or(IndexError::new("meta.tokens", sequence_id))?;
        let start_token = sequence.get(start_token).ok_or(IndexError::new("meta.tokens[]", start_token))?;
        let start_offset = start_token.start();
        let end_token = sequence.get(end_token).ok_or(IndexError::new("meta.tokens[]", end_token))?;
        let end_offset = end_token.end();
        let text = &self.texts.get(sequence_id).ok_or(IndexError::new("meta.texts", sequence_id))?;
        let text = text[start_offset..end_offset].to_string();
        let class = self.entities.get(class).ok_or(IndexError::new("meta.entities", class))?.to_string();
        Ok(Span::new(
            sequence_id, 
            start_offset, 
            end_offset,
            text, 
            class, 
            probability)
        )
    }
}
