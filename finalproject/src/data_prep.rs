use csv::{ReaderBuilder, WriterBuilder};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::error::Error;

#[derive(Debug, Deserialize, Serialize)]
pub struct AirbnbRecord {
    pub neighbourhood_group: Option<String>,
    pub neighbourhood: Option<String>,
    pub room_type: Option<String>,
    pub price: Option<f64>,
    pub minimum_nights: Option<u64>,
    pub number_of_reviews: Option<u64>,
    pub availability_365: Option<u64>,

    // For future use: encoded features
    pub neighbourhood_group_encoded: Option<u8>, 
    pub neighbourhood_encoded: Option<u8>,
    pub room_type_encoded: Option<u8>,
    pub price_category: Option<String>,
}

impl AirbnbRecord {
    pub fn encode_features(&mut self, neighbourhood_group_map: &HashMap<String, u8>, neighbourhood_map: &HashMap<String, u8>) {
        // Encode neighbourhood_group with One-Hot Encoding (if available in the map)
        if let Some(ref group) = self.neighbourhood_group {
            let encoded_group = neighbourhood_group_map.get(group).cloned();
            self.neighbourhood_group_encoded = encoded_group;
        }

        // Encode neighbourhood as an integer (if available in the map)
        if let Some(ref neighbourhood) = self.neighbourhood {
            self.neighbourhood_encoded = neighbourhood_map.get(neighbourhood).cloned();
        }

        // Encode room_type as integers
        if let Some(ref room_type) = self.room_type {
            match room_type.as_str() {
                "Entire home/apt" => self.room_type_encoded = Some(0),
                "Private room" => self.room_type_encoded = Some(1),
                "Shared room" => self.room_type_encoded = Some(2),
                _ => self.room_type_encoded = Some(3), // Default for unknown types
            }
        }

        // Categorize price
        if let Some(price) = self.price {
            self.price_category = Some(if price < 100.0 {
                "low".to_string()
            } else if price < 200.0 {
                "medium".to_string()
            } else {
                "high".to_string()
            });
        }
    }
}

// Helper function to calculate percentiles
fn calculate_percentile(data: &[u64], percentile: f64) -> u64 {
    let mut sorted = data.to_vec();
    sorted.sort_unstable();
    let rank = (percentile / 100.0 * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[rank]
}

pub fn load_and_clean_data(file_path: &str, output_path: &str) -> Result<(), Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().from_path(file_path)?;
    let mut wtr = WriterBuilder::new().from_path(output_path)?;

    // Write headers for the cleaned data
    wtr.write_record(&[
        "neighbourhood_group_encoded", "neighbourhood_encoded", 
        "room_type_encoded", "price_category", "minimum_nights", "number_of_reviews"
    ])?;

    let mut data: Vec<AirbnbRecord> = Vec::new();
    let mut neighbourhood_group_set: HashSet<String> = HashSet::new();
    let mut neighbourhood_set: HashSet<String> = HashSet::new();

    // Step 1: Collect unique neighbourhood_group and neighbourhood values
    for result in rdr.deserialize() {
        let record: AirbnbRecord = result?;
        if let Some(ref group) = record.neighbourhood_group {
            neighbourhood_group_set.insert(group.clone());
        }
        if let Some(ref neighbourhood) = record.neighbourhood {
            neighbourhood_set.insert(neighbourhood.clone());
        }
        data.push(record);
    }

    // Step 2: Create encoding maps from the unique values
    let neighbourhood_group_map: HashMap<String, u8> = neighbourhood_group_set
        .into_iter()
        .enumerate()
        .map(|(index, value)| (value, index as u8))
        .collect();

    let neighbourhood_map: HashMap<String, u8> = neighbourhood_set
        .into_iter()
        .enumerate()
        .map(|(index, value)| (value, index as u8))
        .collect();

    // Step 3: Apply feature encoding to the records
    for record in &mut data {
        record.encode_features(&neighbourhood_group_map, &neighbourhood_map);
    }

    // Step 4: Clean data: Remove outliers based on `availability_365`
    let availabilities: Vec<u64> = data.iter().filter_map(|rec| rec.availability_365).collect();
    let q1 = calculate_percentile(&availabilities, 25.0);
    let q3 = calculate_percentile(&availabilities, 75.0);
    let iqr = (q3 - q1) as f64;
    let lower_bound = (q1 as f64 - iqr * 1.5).max(0.0) as u64;
    let upper_bound = (q3 as f64 + iqr * 1.5).max(0.0) as u64;

    // Step 5: Remove records with availability outside the computed bounds
    data.retain(|rec| match rec.availability_365 {
        Some(value) => value >= lower_bound && value <= upper_bound,
        None => false,
    });

    // Step 6: Write cleaned and encoded data to the output file
    for record in data {
        // Skip records with missing 'price' or 'neighbourhood_group'
        if record.price.is_none() || record.neighbourhood_group.is_none() {
            continue;
        }

        // Write only the encoded data, no need for the neighbourhood name
        wtr.write_record(&[
            record.neighbourhood_group_encoded.unwrap_or_default().to_string(),  
            record.neighbourhood_encoded.unwrap_or_default().to_string(),
            record.room_type_encoded.unwrap_or_default().to_string(),
            record.price_category.unwrap_or_default(),
            record.minimum_nights.unwrap_or_default().to_string(),
            record.number_of_reviews.unwrap_or_default().to_string(),
        ])?;
    }

    Ok(())
}
