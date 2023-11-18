use anyhow::Result;
use clap::{Parser, Subcommand};

mod cli;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Convert fined tuned Stable Diffusion version into burn's native format
    Convert {
        #[clap(flatten)]
        args: cli::convert::ConvertArgs,
    },
    /// Sample from a fined tuned Stable Diffusion model and save the results
    Sample {
        #[clap(flatten)]
        args: cli::txt2img::Txt2ImgArgs,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Convert { args } => {
            cli::convert::handle_convert(args)?;
        }
        Commands::Sample { args } => {
            cli::txt2img::handle_txt2img(args)?;
        }
    }

    Ok(())
}