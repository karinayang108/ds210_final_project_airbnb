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
    let max_depths: Vec<usize> = (3..=50).collect();
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


#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{File, remove_file};
    use std::io::Write;
    use std::path::Path;

    #[test]
    fn test_data_pipeline() {
        // Test CSV file creation and processing
        let test_input = "neighbourhood_group,room_type,price,minimum_nights,number_of_reviews,availability_365\n\
                          Brooklyn,Entire home/apt,150.0,2,10,365\n\
                          Manhattan,Private room,100.0,3,50,180\n\
                          Queens,Shared room,50.0,1,5,365";

        let test_file_path = "test_data.csv";
        let output_file_path = "test_cleaned_data.csv";

        // Create test file
        File::create(test_file_path)
            .unwrap()
            .write_all(test_input.as_bytes())
            .unwrap();

        // Test data loading and cleaning
        assert!(load_and_clean_data(test_file_path, output_file_path).is_ok());
        assert!(Path::new(output_file_path).exists());

        // Clean up test files
        remove_file(test_file_path).unwrap();
        remove_file(output_file_path).unwrap();
    }

    #[test]
    fn test_machine_learning_pipeline() {
        // Sample data for preprocessing and model training
        let records = vec![
            AirbnbCleanedRecord {
                neighbourhood_group_encoded: 1,
                room_type_encoded: 0,
                price_category: "low".to_string(),
                minimum_nights: 2,
                number_of_reviews: 10,
            },
            AirbnbCleanedRecord {
                neighbourhood_group_encoded: 2,
                room_type_encoded: 1,
                price_category: "medium".to_string(),
                minimum_nights: 3,
                number_of_reviews: 5,
            },
        ];

        // Test preprocessing
        let (features, targets) = preprocess_data(&records);
        assert!(!features.is_empty());
        assert!(!targets.is_empty());
        assert_eq!(features.nrows(), records.len());
        assert_eq!(features.ncols(), 4);
        assert_eq!(targets.len(), records.len());

        // Test decision tree training
        let decision_tree = train_decision_tree(&features, &targets, Some(3), 0.01);
        
        // Test feature importance
        let feature_importance = get_decision_tree_feature_importance(&decision_tree);
        assert!(!feature_importance.is_empty());
        
        // Validate model performance
        let accuracy = evaluate_decision_tree(&decision_tree, &features, &targets);
        assert!(accuracy > 0.0);
    }
}