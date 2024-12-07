mod decision_tree;
use decision_tree::*;
mod data_prep;
use crate::data_prep::load_and_clean_data;

fn main() {
    let raw_file = "AB_NYC_2019.csv";
    let cleaned_file = "cleaned_file.csv";

    // Step 1: Load and clean raw data
    let _=load_and_clean_data(raw_file, cleaned_file);

    // Step 2: Load and preprocess cleaned data
    let records = process_csv_file(cleaned_file);

    // Split data into 80% training and 20% testing
    let (train_records, test_records) = split_data(records, 0.8);

    println!("Training data size: {}", train_records.len());
    println!("Testing data size: {}", test_records.len());

    let (train_features, train_targets) = preprocess_data(&train_records);
    let (test_features, test_targets) = preprocess_data(&test_records);

    // Train decision tree classifier
    let decision_tree = train_decision_tree(&train_features, &train_targets);
    println!("Decision tree successfully trained.");

    // Evaluate the decision tree on the test dataset
    let accuracy = evaluate_decision_tree(&decision_tree, &test_features, &test_targets);
    println!("Test accuracy: {}", accuracy);

    // Export decision tree visualization
    export_decision_tree(&decision_tree);
}
