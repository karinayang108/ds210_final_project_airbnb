use linfa_trees::DecisionTree;
use std::collections::HashMap;

/// Function to retrieve the feature importance of a pre-trained decision tree
pub fn get_decision_tree_feature_importance(decision_tree: &DecisionTree<f64, usize>) -> Vec<(usize, f64)> {
    let feature_importance = decision_tree.feature_importance();

    // Check if feature importance is empty and return an empty list if so
    if feature_importance.is_empty() {
        eprintln!("Warning: The decision tree does not have feature importance values.");
        return Vec::new();
    }
    // Map the feature importance to a Vec<(feature_index, importance_value)>
    let mut feature_with_importance: Vec<(usize, f64)> = feature_importance.iter()
        .enumerate()
        .map(|(idx, &importance)| (idx, importance))
        .collect();

    // Sort the feature importance in descending order based on importance value
    feature_with_importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    feature_with_importance
}

/// Function to retrieve the feature names for the dataset
/// Returns a HashMap mapping feature indices to feature names
pub fn get_feature_names() -> HashMap<usize, &'static str> {
    let mut feature_names = HashMap::new();
    
    // Define feature names corresponding to their indices
    feature_names.insert(0, "neighbourhood_encoded");
    feature_names.insert(1, "room_type_encoded");
    feature_names.insert(2, "minimum_nights");
    feature_names.insert(3, "number_of_reviews");

    feature_names
}


