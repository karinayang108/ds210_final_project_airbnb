mod decision_tree;
use decision_tree::*;
mod data_prep;
use data_prep::load_and_clean_data;
mod further_eval;
use further_eval::*;

fn main() {
    let raw_file = "AB_NYC_2019.csv";
    let cleaned_file = "cleaned_file.csv";
    // Step 1: Load and clean raw data
    let _ =load_and_clean_data(raw_file, cleaned_file);
    // Step 2: Load and preprocess cleaned data
    let records = process_csv_file(cleaned_file);
    // Split data into 80% training and 20% testing
    println!("Splitting data into training and testing sets...");
    let (train_records, test_records) = split_data(records, 0.8);
    println!("Training data size: {}", train_records.len());
    println!("Testing data size: {}", test_records.len());

    // Preprocess data
    println!("Preprocessing training and testing data...");
    let (train_features, train_targets) = preprocess_data(&train_records);
    let (test_features, test_targets) = preprocess_data(&test_records);

    // Find the best max_depth for the decision tree
    println!("Finding the best max_depth...");
    let max_depths = vec![3, 5, 10, 15, 20];
    let best_max_depth = find_best_max_depth(&train_features, &train_targets, &max_depths);
    println!("Best max_depth found: {}", best_max_depth);

    // Train decision tree classifier with the best max_depth
    println!("Training decision tree with max depth: {}", best_max_depth);
    let decision_tree = train_decision_tree(&train_features, &train_targets, Some(best_max_depth), 0.01);
    println!("Decision tree successfully trained.");

    // Evaluate the decision tree on the test dataset
    println!("Evaluating the decision tree...");
    let accuracy = evaluate_decision_tree(&decision_tree, &test_features, &test_targets);
    println!("Test accuracy: {:.2}%", accuracy);

    // Export decision tree visualization
    println!("Exporting decision tree visualization...");
    export_decision_tree(&decision_tree);
    compile_to_pdf();

    println!("Calculating feature importance based on the decision tree...");
    let feature_importance = get_decision_tree_feature_importance(&decision_tree);
    let feature_names = get_feature_names();

    println!("Feature importance based on the decision tree:");
    // Safely iterate through the feature importance
    for (index, importance) in feature_importance {
        // Use `unwrap_or` to safely handle missing feature names
        let feature_name = feature_names.get(&index).unwrap_or(&"Unknown Feature");
        println!("Feature {}: {} - Importance: {:.4}", index, feature_name, importance);
    }
}

