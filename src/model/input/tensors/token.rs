use ort::session::SessionInputs;
use crate::util::compose::Composable;
use crate::util::result::Result;
use super::super::encoded::EncodedInput;
use super::super::super::pipeline::context::TensorsMeta;

/// Ready-for-inference tensors (token mode)
pub struct TokenTensors<'a> {
    pub tensors: SessionInputs<'a, 'a>,
    pub meta: TensorsMeta,    
}

impl<'a> TokenTensors<'a> {

    pub fn from(encoded: EncodedInput) -> Result<Self> {
        let inputs = ort::inputs!{
            "input_ids" => encoded.input_ids,
            "attention_mask" => encoded.attention_masks,
            "words_mask" => encoded.word_masks,
            "text_lengths" => encoded.text_lengths,
        }?;
        Ok(Self {
            tensors: inputs.into(),
            meta: TensorsMeta { 
                texts: encoded.texts, 
                tokens: encoded.tokens, 
                entities: encoded.entities, 
                num_words: encoded.num_words 
            },            
        })
    }

}


/// Composable: Encoded => TokenTensors
#[derive(Default)]
pub struct EncodedToTensors { }


impl<'a> Composable<EncodedInput, TokenTensors<'a>> for EncodedToTensors {
    fn apply(&self, input: EncodedInput) -> Result<TokenTensors<'a>> {
        TokenTensors::from(input)
    }
}


/// Composable: TokenTensors => (SessionInput, TensorsMeta) 
#[derive(Default)]
pub struct TensorsToSessionInput { }


impl<'a> Composable<TokenTensors<'a>, (SessionInputs<'a, 'a>, TensorsMeta)> for TensorsToSessionInput {
    fn apply(&self, input: TokenTensors<'a>) -> Result<(SessionInputs<'a, 'a>, TensorsMeta)> {
        Ok((input.tensors, input.meta))
    }
}
