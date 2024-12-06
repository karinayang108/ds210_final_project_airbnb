mod data_prep;
use crate::data_prep::load_and_clean_data;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load, clean, and save the data
    load_and_clean_data("AB_NYC_2019.csv", "cleaned_file.csv")?;
    Ok(())
}
