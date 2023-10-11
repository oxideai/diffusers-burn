use anyhow::Result;
use clap::Args;

#[derive(Args, Debug)]
pub struct ConvertArgs {
    /// Path to the fined tuned model
    input: String,

    /// Output directory to save the converted model
    #[arg(short)]
    output_dir: String,
}

pub fn handle_convert(args: &ConvertArgs) -> Result<()> {
    Ok(())
}
