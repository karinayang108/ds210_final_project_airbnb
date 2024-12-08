use csv::{ReaderBuilder, WriterBuilder};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::error::Error;

#[derive(Debug, Deserialize, Serialize)]
pub struct AirbnbRecord {
    pub neighbourhood_group: Option<String>,
    pub room_type: Option<String>,
    pub price: Option<f64>,
    pub minimum_nights: Option<u64>,
    pub number_of_reviews: Option<u64>,
    pub availability_365: Option<u64>,

    pub neighbourhood_group_encoded: Option<u8>, 
    pub room_type_encoded: Option<u8>,
    pub price_category: Option<String>,
}

impl AirbnbRecord {
    pub fn encode_features(
        &mut self,
        neighbourhood_group_map: &HashMap<String, u8>, 
        price_percentiles: &[f64]
    ) {
        // Encode neighbourhood_group
        if let Some(ref group) = self.neighbourhood_group {
            self.neighbourhood_group_encoded = neighbourhood_group_map.get(group).cloned();
        }

        // Encode room_type
        if let Some(ref room_type) = self.room_type {
            self.room_type_encoded = match room_type.as_str() {
                "Entire home/apt" => Some(0),
                "Private room" => Some(1),
                "Shared room" => Some(2),
                _ => Some(3),
            };
        }

        // Categorize price based on percentiles
        if let Some(price) = self.price {
            self.price_category = if price < price_percentiles[0] {
                Some("low".to_string())
            } else if price < price_percentiles[1] {
                Some("medium".to_string())
            } else {
                Some("high".to_string())
            };
        }
    }
}

fn calculate_percentile(data: &[f64], percentile: f64) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)); 
    let rank = (percentile / 100.0 * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[rank]
}

pub fn load_and_clean_data(file_path: &str, output_path: &str) -> Result<(), Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().from_path(file_path)?;
    let mut wtr = WriterBuilder::new().from_path(output_path)?;

    // Write headers for the cleaned data
    wtr.write_record(&[
        "neighbourhood_group_encoded","room_type_encoded", "price_category", "minimum_nights", "number_of_reviews"
    ])?;

    let mut data: Vec<AirbnbRecord> = Vec::new();
    let mut neighbourhood_group_set = HashSet::new();
    let mut prices = Vec::new();
    let mut min_nights = Vec::new();
    let mut num_reviews = Vec::new();
    let mut availabilities = Vec::new();

    for result in rdr.deserialize() {
        let record: AirbnbRecord = result?;
        if let Some(ref group) = record.neighbourhood_group {
            neighbourhood_group_set.insert(group.clone());
        }
        if let Some(price) = record.price {
            prices.push(price);
        }
        if let Some(min_night) = record.minimum_nights {
            min_nights.push(min_night as f64);
        }
        if let Some(num_review) = record.number_of_reviews {
            num_reviews.push(num_review as f64);
        }
        if let Some(availability) = record.availability_365 {
            availabilities.push(availability as f64);
        }
        data.push(record);
    }

    // Calculate IQR and bounds for outlier detection
    let calc_iqr_and_bounds = |data: &[f64]| -> (f64, f64) {
        let q1 = calculate_percentile(data, 25.0);
        let q3 = calculate_percentile(data, 75.0);
        let iqr = q3 - q1;
        let lower_bound = (q1 - 1.5 * iqr).max(0.0);
        let upper_bound = q3 + 1.5 * iqr;
        (lower_bound, upper_bound)
    };

    let (min_nights_lower, min_nights_upper) = calc_iqr_and_bounds(&min_nights);
    let (num_reviews_lower, num_reviews_upper) = calc_iqr_and_bounds(&num_reviews);
    let (price_lower, price_upper) = calc_iqr_and_bounds(&prices);
    let (availability_lower, availability_upper) = calc_iqr_and_bounds(&availabilities);

    // Filter data for valid ranges
    data.retain(|rec| {
        rec.minimum_nights.map_or(false, |min| (min as f64) >= min_nights_lower && (min as f64) <= min_nights_upper) &&
        rec.number_of_reviews.map_or(false, |num| (num as f64) >= num_reviews_lower && (num as f64) <= num_reviews_upper) &&
        rec.price.map_or(false, |price| price >= price_lower && price <= price_upper) &&
        rec.availability_365.map_or(false, |avail| (avail as f64) >= availability_lower && (avail as f64) <= availability_upper)
    });

    // Create encoding maps
    let neighbourhood_group_map: HashMap<String, u8> = neighbourhood_group_set
        .into_iter()
        .enumerate()
        .map(|(index, value)| (value, index as u8))
        .collect();

    // Calculate price percentiles
    let price_percentiles = vec![
        calculate_percentile(&prices, 33.0),
        calculate_percentile(&prices, 66.0),
    ];

    // Encode features for each record
    for record in &mut data {
        record.encode_features(&neighbourhood_group_map, &price_percentiles);
    }

    // Write cleaned and encoded data
    for record in data {
        if record.price.is_none() || record.neighbourhood_group.is_none() {
            continue;
        }
        wtr.write_record(&[
            record.neighbourhood_group_encoded.unwrap_or_default().to_string(),
            record.room_type_encoded.unwrap_or_default().to_string(),
            record.price_category.clone().unwrap_or_default(),
            record.minimum_nights.unwrap_or_default().to_string(),
            record.number_of_reviews.unwrap_or_default().to_string(),
        ])?;
    }

    Ok(())
}
