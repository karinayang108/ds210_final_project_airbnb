use std::collections::HashMap;
use ndarray::{Array2};

pub fn heuristic_feature_importance(features: &Array2<f64>) -> Vec<(usize, f64)> {
    let mut importance: HashMap<usize, f64> = HashMap::new();

    for feature_idx in 0..features.shape()[1] {
        let column = features.column(feature_idx);
        let variance = column.var(0.0); 
        importance.insert(feature_idx, variance);
    }

    // Map indices and sort them by importance
    let mut feature_with_importance: Vec<(usize, f64)> = importance.into_iter().collect();
    // Sort features by computed variance (importance) descending
    feature_with_importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    feature_with_importance
}


pub fn get_feature_names() -> HashMap<usize, &'static str> {
    let mut feature_names = HashMap::new();
    feature_names.insert(0, "neighbourhood_group_encoded");
    feature_names.insert(1, "neighbourhood_encoded");
    feature_names.insert(2, "latitude");
    feature_names.insert(3, "longitude");
    feature_names.insert(4, "room_type_encoded");
    feature_names.insert(5, "minimum_nights");
    feature_names.insert(6, "number_of_reviews");

    feature_names
}

