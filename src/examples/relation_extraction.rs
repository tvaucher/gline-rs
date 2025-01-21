use gliner::model::input::relation::schema::RelationSchema;
use gliner::model::pipeline::relation::RelationPipeline;
use gliner::model::pipeline::token::TokenPipeline;
use gliner::util::result::Result;
use gliner::model::{input::text::TextInput, params::{Parameters, RuntimeParameters}};


/// Sample usage of the public API for Relation Extraction
/// 
/// Also provides an example of advanced use of `Model` and `Pipeline`.
fn main() -> Result<()> {

    // paths to the model and tokenizer
    const MODEL_PATH: &str = "models/gliner-multitask-large-v0.5/onnx/model.onnx";
    const TOKENIZER_PATH: &str = "models/gliner-multitask-large-v0.5/tokenizer.json";    

    // define a sample input
    let input = TextInput::from_str(
        &[ 
            "Bill Gates is an American businessman who co-founded Microsoft."
        ],
        &[
            "person", 
            "company",            
        ]
    )?;

    // define a relation schema with a "founded" relation which subject has to be a "person" and object has to be a "company"
    let mut relation_schema = RelationSchema::new();
    relation_schema.push_with_allowed_labels("founded", &["person"], &["company"]);
    
    // load de model
    println!("Loading model...");  
    let runtime_params =   RuntimeParameters::default();
    let model = gliner::model::inference::Model::new(MODEL_PATH, runtime_params)?;
    
    // load two pipelines: one for NER, and one for Relation Extraction
    let params = Parameters::default();
    let ner_pipeline = TokenPipeline::new(TOKENIZER_PATH)?;
    let rel_pipeline = RelationPipeline::default(TOKENIZER_PATH, relation_schema)?;

    // first, perform entity extraction using the NER pipeline
    println!("Inferencing (NER)...");
    let ner_output = model.inference(input, &ner_pipeline, &params)?;

    // print results
    for spans in &ner_output.spans {
        for span in spans {
            println!("{:3} | {:15} | {:10} | {:.1}%", span.sequence(), span.text(), span.class(), span.probability() * 100.0);
        }
    }
    
    // then, perform relation extraction using the RE pipeline, leveraging the output from NER
    println!("Inferencing (RE)...");
    let rel_output = model.inference(ner_output, &rel_pipeline, &params)?;
    
    // print results
    for relations in &rel_output.relations {
        for relation in relations {
            println!("{:3} | {:15} | {:10} | {:15} | {:.1}%", relation.sequence(), relation.subject(), relation.class(), relation.object(), relation.probability() * 100.0);
        }
    }

    Ok(())

}
