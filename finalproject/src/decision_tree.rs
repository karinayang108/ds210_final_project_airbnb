use ndarray::{Array2, Array1};
use linfa_trees::DecisionTree;
use linfa::prelude::*;
use linfa::traits::Predict;
use serde::Deserialize;
use std::fs::File;
use std::io::Write;
use rand::seq::SliceRandom;
use std::process::Command;

// Deserialize CSV fields into a struct matching your dataset
#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "lowercase")]
pub struct AirbnbCleanedRecord {
    pub neighbourhood_group_encoded: u8,
    pub neighbourhood_encoded: u8,
    pub room_type_encoded: u8,
    pub price_category: String,
    pub minimum_nights: u64,
    pub number_of_reviews: u64,
}

// Function to process CSV data into usable Vec<AirbnbCleanedRecord>
pub fn process_csv_file(file_path: &str) -> Vec<AirbnbCleanedRecord> {
    let mut rdr = csv::Reader::from_path(file_path).unwrap();
    let mut v: Vec<AirbnbCleanedRecord> = Vec::new();
    
    for result in rdr.deserialize() {
        let record: AirbnbCleanedRecord = result.expect("Error parsing CSV record");
        v.push(record);
    }
    
    println!("Number of records read from CSV: {}", v.len());
    v
}

// Function to split the data into train and test sets
pub fn split_data(records: Vec<AirbnbCleanedRecord>, train_ratio: f64) -> (Vec<AirbnbCleanedRecord>, Vec<AirbnbCleanedRecord>) {
    let mut rng = rand::thread_rng();
    let mut records = records.clone();
    records.shuffle(&mut rng); // Shuffle the data randomly
    let split_index = (records.len() as f64 * train_ratio) as usize;
    let train_set = records[..split_index].to_vec();
    let test_set = records[split_index..].to_vec();
    (train_set, test_set)
}

pub fn normalize_features(features: &mut Array2<f64>) {
    for col in 0..features.ncols() {
        let col_view = features.column(col);
        let mean = col_view.mean().unwrap_or(0.0);
        let std = col_view.std(0.0);
        
        if std > 0.0 {
            features.column_mut(col).mapv_inplace(|x| (x - mean) / std);
        }
    }
}

// Preprocess data into features and targets
pub fn preprocess_data(records: &[AirbnbCleanedRecord]) -> (Array2<f64>, Array1<usize>) {
    let mut features = Vec::with_capacity(records.len() * 5);
    let mut targets = Vec::with_capacity(records.len());

    for record in records {
        features.push(record.neighbourhood_group_encoded as f64);
        features.push(record.neighbourhood_encoded as f64);
        features.push(record.room_type_encoded as f64);
        features.push(record.minimum_nights as f64);
        features.push(record.number_of_reviews as f64);

        // More explicit target encoding
        let target = match record.price_category.as_str() {
            "high" => 2,
            "medium" => 1,
            "low" => 0,
            _ => panic!("Unexpected price category")
        };
        targets.push(target);
    }

    let mut features = Array2::from_shape_vec((records.len(), 5), features)
        .expect("Error creating feature matrix");
    let targets = Array1::from(targets);
    normalize_features(&mut features);

    (features, targets)
}

// Train decision tree
pub fn train_decision_tree(features: &Array2<f64>, targets: &Array1<usize>, max_depth: Option<usize>, min_impurity_decrease: f64) -> DecisionTree<f64, usize> {
    let dataset = Dataset::new(features.clone(), targets.clone());

    DecisionTree::params()
        .max_depth(max_depth)
        .min_impurity_decrease(min_impurity_decrease)
        .fit(&dataset)
        .expect("Error fitting DecisionTree")
}

pub fn find_best_max_depth(
    features: &Array2<f64>, 
    targets: &Array1<usize>, 
    depths: &[usize]
) -> usize {
    let mut best_depth = 0;
    let mut best_accuracy = 0.0;

    for &depth in depths {
        let decision_tree = train_decision_tree(features, targets, Some(depth), 0.01);
        let accuracy = evaluate_decision_tree(&decision_tree, features, targets);

        if accuracy > best_accuracy {
            best_depth = depth;
            best_accuracy = accuracy;
        }
    }

    println!("Best max depth: {}, Accuracy: {:.2}%", best_depth, best_accuracy);
    best_depth
}

// Evaluate decision tree
pub fn evaluate_decision_tree(
    decision_tree: &DecisionTree<f64, usize>, 
    test_features: &Array2<f64>, 
    test_targets: &Array1<usize>
) -> f64 {
    let test_dataset = Dataset::new(test_features.clone(), test_targets.clone());
    
    // Use predict and calculate accuracy manually
    let predictions = decision_tree.predict(&test_dataset);
    let correct_predictions = predictions
        .iter()
        .zip(test_targets.iter())
        .filter(|&(pred, actual)| pred == actual)
        .count();
    
    (correct_predictions as f64 / test_targets.len() as f64) * 100.0
}

// Export decision tree visualization
pub fn export_decision_tree(decision_tree: &DecisionTree<f64, usize>) {
    let mut tikz = File::create("decision_tree_example.tex").unwrap();
    tikz.write_all(
        decision_tree
            .export_to_tikz()
            .with_legend()
            .to_string()
            .as_bytes(),
    )
    .unwrap();
    println!("Decision tree visualization exported to decision_tree_example.tex!");
}


pub fn compile_to_pdf() {
    let output = Command::new("pdflatex")
        .arg("decision_tree_example.tex")
        .output()
        .expect("Failed to execute pdflatex");

    if output.status.success() {
        println!("PDF successfully generated: decision_tree.pdf");
    } else {
        eprintln!("Failed to compile LaTeX file. Output: {:?}", output);
    }
}