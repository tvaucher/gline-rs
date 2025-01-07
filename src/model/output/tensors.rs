//! Encapsulation of raw tensor outputs

use ort::session::SessionOutputs;
use crate::util::{result::Result, compose::Composable};
use crate::model::pipeline::TensorsMeta;

/// Represents the raw tensor output of the inference step
pub struct TensorOutput<'a> {
    pub meta: TensorsMeta,
    pub tensors: SessionOutputs<'a, 'a>,
}


impl<'a> TensorOutput<'a> {
    pub fn from(tensors: SessionOutputs<'a, 'a>, meta: TensorsMeta) -> Self {
        Self { 
            meta,
            tensors 
        }
    }
}


/// Composable: (SessionOutput, TensorMeta) => TensorOutput
pub struct SessionOutputToTensors { }

impl SessionOutputToTensors {
    pub fn new() -> Self { Self {} }
}

impl<'a> Composable<(SessionOutputs<'a, 'a>, TensorsMeta), TensorOutput<'a>> for SessionOutputToTensors {
    fn apply(&self, input: (SessionOutputs<'a, 'a>, TensorsMeta)) -> Result<TensorOutput<'a>> {
        Ok(TensorOutput::from(input.0, input.1))
    }
}
