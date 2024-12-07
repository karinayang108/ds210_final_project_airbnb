use linfa_trees::DecisionTree;
use std::collections::HashMap;

pub fn get_decision_tree_feature_importance(
    decision_tree: &DecisionTree<f64, usize>
) -> Vec<(usize, f64)> {
    // Get feature importance from the pre-trained decision tree
    let feature_importance = decision_tree.feature_importance();

    // Map the feature importance to a Vec<(feature_index, importance_value)>
    let mut feature_with_importance: Vec<(usize, f64)> = feature_importance.iter()
        .enumerate()
        .map(|(idx, &importance)| (idx, importance))
        .collect();

    // Sort the feature importance in descending order
    feature_with_importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    feature_with_importance
}

pub fn get_feature_names() -> HashMap<usize, &'static str> {
    let mut feature_names = HashMap::new();
    feature_names.insert(0, "neighbourhood_group_encoded");
    feature_names.insert(1, "neighbourhood_encoded");
    feature_names.insert(2, "room_type_encoded");
    feature_names.insert(3, "minimum_nights");
    feature_names.insert(4, "number_of_reviews");

    feature_names
}