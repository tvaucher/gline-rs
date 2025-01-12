use gliner::util::result::Result;
use gliner::model::{input::text::TextInput, params::{Parameters, RuntimeParameters}, GLiNER};
use gliner::model::pipeline::token::TokenMode;
use ort::execution_providers::{CUDAExecutionProvider, CoreMLExecutionProvider, DirectMLExecutionProvider};


/// Sample usage of the public API using GPU for inferencing
/// 
/// This example will try most common execution providers in that order:
/// * DirectML
/// * CUDA
/// * CoreML
/// 
/// To leverage one of them you need to enable the appropriate feature, for exemple:
/// 
/// ```console
/// $ cargo run --example gpu --features=cuda
/// ```
/// 
/// See <https://ort.pyke.io/perf/execution-providers>
fn main() -> Result<()> {    

    const MAX_SAMPLES: usize = 100;
    const CSV_PATH: &str = "data/nuner-sample-1k.csv";

    let entities = [
        "person", 
        "location",
        "vehicle",
    ];

    println!("Loading data...");
    let input = TextInput::new_from_csv(CSV_PATH, 0, MAX_SAMPLES, entities.map(|x| x.to_string()).to_vec())?;
    let nb_samples = input.texts.len();
    
    println!("Loading model...");
    let model = GLiNER::<TokenMode>::new(
        Parameters::default(),
        RuntimeParameters::default().with_execution_providers([
            DirectMLExecutionProvider::default().build(),
            CUDAExecutionProvider::default().build(),
            CoreMLExecutionProvider::default().build(),            
        ]),
        "models/gliner-multitask-large-v0.5/tokenizer.json",
        "models/gliner-multitask-large-v0.5/onnx/model.onnx",
    )?;

    println!("Inferencing...");
    let inference_start = std::time::Instant::now();
    let _output = model.inference(input)?;
    
    let inference_time = inference_start.elapsed();
    println!("Inference took {} seconds on {} samples ({:.2} sec/sample)", inference_time.as_secs(), nb_samples, inference_time.as_secs() as f32 / nb_samples as f32);

    Ok(())
}

