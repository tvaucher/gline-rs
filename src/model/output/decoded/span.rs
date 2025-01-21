//! First step of span decoding (in span mode)

use crate::util::math::sigmoid;
use crate::util::result::Result;
use crate::util::compose::Composable;
use crate::text::span::Span;
use crate::model::pipeline::context::TensorsMeta;
use crate::model::output::tensors::TensorOutput;
use super::SpanOutput;


/// Decoding method for span mode.
/// 
/// See sections 2.1 and 2.3 of the [original paper](https://arxiv.org/abs/2311.08526).
/// Note: greedy search is not included in this step and must be applied subsequently.
pub struct TensorsToDecoded {
    threshold: f32,
    max_width: usize,
}

impl TensorsToDecoded {
    pub fn new(threshold: f32, max_width: usize) -> Self {
        Self { 
            threshold,
            max_width,
        }
    }

    fn decode(&self, input: &TensorOutput) -> Result<Vec<Vec<Span>>> {        
        // prepare output vector
        let batch_size = input.meta.texts.len();
        let mut result: Vec<Vec<Span>> = Vec::new();

        // look for logits and check its shape
        let logits = input.tensors.get("logits").ok_or("logits not found in model output")?;
        self.check_shape(logits.shape()?, &input.meta)?;
        
        // extract the actual array
        let array = logits.try_extract_tensor::<f32>()?;

        // iterate over the sequences
        for sequence_id in 0..batch_size {
            // get a slice for the current sequence (1st dimension)
            let sequence = array.slice(ndarray::s![sequence_id, .., .., ..]);
            let num_tokens = input.meta.tokens.get(sequence_id).unwrap().len();
            //println!("{:?}", sequence.map(|x| crate::util::math::sigmoid(*x)));
            
            // prepare the list of spans for this sequence
            let mut spans = Vec::new();
            
            // iterate over all spans
            for ((start, end, class), score) in sequence.indexed_iter() {
                // check that the tokens actually exist in the current sequence (we could do better here, to avoid iterating over these ones)                
                if start >= num_tokens || start + end >= num_tokens {
                    continue;
                }
                // check that the score is above threshold (otherwise continue)
                let score = sigmoid(*score);
                if score >= self.threshold {
                    // if yes, create the span
                    spans.push(input.meta.create_span(sequence_id, start, start+end, class, score)?);
                }
            }
            
            // add the list of spans for this sequence
            result.push(spans);
        }
        
        // return
        Ok(result)
    }


    /// Checks coherence of the output shape
    /// Expected shape is (batch_size, num_words, num_spans, num_classes)
    fn check_shape(&self, actual_shape: Vec<i64>, meta: &TensorsMeta) -> Result<()> {
        let expected_shape = vec![meta.texts.len() as i64, meta.num_words as i64, self.max_width as i64, meta.entities.len() as i64];
        if actual_shape != expected_shape {
            Err("unexpected logits shape".into())
        }
        else {
            Ok(())
        }
    }

}

impl<'a> Composable<TensorOutput<'a>, SpanOutput> for TensorsToDecoded {
    fn apply(&self, input: TensorOutput) -> Result<SpanOutput> {        
        let decoded = self.decode(&input)?;
        Ok(SpanOutput::new(input.meta.texts, input.meta.entities, decoded))
    }
}