//! Processing and inferencing parameters

use ort::execution_providers::ExecutionProviderDispatch;


/// Represents the set of parameters for the whole pipeline
/// 
/// * pre-processing
/// * post-processing
/// 
/// The easiest way to instanciate sound parameters is to use the
/// `default()` constructor and then use individual setters as needed.
pub struct Parameters {
    /// Probability threshold (default: 0.5)
    pub threshold: f32,    
    /// Flat NER means that an entity can not embed another one (default: true)
    pub flat_ner: bool,
    /// Multi-label means that the same span can belong to multiple classes (default: false)
    pub multi_label: bool,    
    /// For span mode, maximum span width (default: 12)
    pub max_width: usize,
    /// Maximum sequence length (default: 512)
    pub max_length: Option<usize>,
}

impl Default for Parameters {
    /// Default configuration, which can be safely used in most cases
    fn default() -> Self {
        Self::new(
            0.5, 
            12, 
            Some(512),
            true, 
            false,
        )
    }
}

impl Parameters {
    /// New configuration specifying every parameter
    pub fn new(threshold: f32, max_width: usize, max_length: Option<usize>, flat_ner: bool, multi_label: bool) -> Self {
        Self { 
            threshold, 
            max_width, 
            max_length,
            flat_ner,
            multi_label,
        }
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    pub fn with_max_width(mut self, max_width: usize) -> Self {
        self.max_width = max_width;
        self
    }

    pub fn with_max_length(mut self, max_length: Option<usize>) -> Self {
        self.max_length = max_length;
        self
    }

    pub fn with_flat_ner(mut self, flat_ner: bool) -> Self {
        self.flat_ner = flat_ner;
        self
    }

    pub fn with_multi_label(mut self, multi_label: bool) -> Self {
        self.multi_label = multi_label;
        self
    }

}


/// Represents the set of parameters for the inference engine
/// 
/// The easiest way to instanciate sound parameters is to use the
/// `default()` constructor and then use individual setters as needed.
pub struct RuntimeParameters {
    /// Number ot threads (default: 4)
    pub threads: usize,
    /// Execution providers (default: none (-> CPU))
    pub execution_providers: Vec<ExecutionProviderDispatch>,
}

impl RuntimeParameters {
    pub fn new(threads: usize, execution_providers: impl IntoIterator<Item = ExecutionProviderDispatch>) -> Self {
        Self {
            threads,
            execution_providers: execution_providers.into_iter().collect(),
        }
    }

    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    pub fn with_execution_providers(mut self, execution_providers: impl IntoIterator<Item = ExecutionProviderDispatch>) -> Self {
        self.execution_providers = execution_providers.into_iter().collect();
        self
    }
}

impl Default for RuntimeParameters {
    fn default() -> Self {
        Self::new(4, [])
    }
}