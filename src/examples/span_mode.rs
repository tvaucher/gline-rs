use gliner::util::result::Result;
use gliner::model::{input::text::TextInput, params::Parameters, GLiNER};
use gliner::model::pipeline::span::SpanMode;

/// Sample usage of the public API in span mode
fn main() -> Result<()> {    
    
    println!("Loading model...");
    let model = GLiNER::<SpanMode>::new(
        Parameters::default(),
        "models/gliner_small-v2.1/tokenizer.json",
        "models/gliner_small-v2.1/onnx/model.onnx",
    )?;

    let input = TextInput::from_str(
        &[ 
            "I am James Bond",
            "This is James and I live in Chelsea, London.",
            "My name is Bond, James Bond.",
            "I like to drive my Aston Martin."
        ],
        &[
            "person", 
            "location",
            "vehicle",
        ]
    )?;

    println!("Inferencing...");
    let output = model.inference(input)?;

    println!("Results:");
    for spans in output.spans {
        for span in spans {
            println!("{:3} | {:15} | {:10} | {:.1}%", span.sequence(), span.text(), span.class(), span.probability() * 100.0);
        }
    }

    Ok(())

}
