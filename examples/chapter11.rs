use std::cmp::{max, min, Ordering};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::iter::FromIterator;
use std::ops::Mul;

use datasets::text::imdb_reviews;
use datasets::Dataset;
use indicatif::{ProgressBar, ProgressStyle};
use rand::distributions::Standard;
use rulinalg::matrix::{BaseMatrix, Matrix};

use grokking_deep_learning_rs::{generate_random_vector, sigmoid_mut};

fn main() -> Result<(), Box<dyn Error>> {
    let (train_dataset, test_dataset) = imdb_reviews()?;
    let train_dataset_size = 2000;
    let test_dataset_size = 2000;

    let (train_reviews, train_labels): (Vec<_>, Vec<_>) = train_dataset
        .shuffle(25000, 0)
        .map(|(s, l): (String, u8)| (s.to_lowercase(), l))
        // currently only considering alphabets and nothing else.
        .map(|(s, l)| {
            (
                s.chars()
                    .map(|c| if c >= 'a' && c <= 'z' { c } else { ' ' })
                    .collect(),
                l,
            )
        })
        .take(train_dataset_size)
        .unzip();

    let (test_reviews, test_labels): (Vec<String>, Vec<_>) = test_dataset
        .shuffle(25000, 0)
        .map(|(s, l)| (s.to_lowercase(), l))
        .take(test_dataset_size)
        .unzip();

    // can't immutably borrow here
    let words = train_reviews
        .iter()
        .flat_map(|s: &String| s.split_whitespace().filter(|w| w.len() > 0));

    let words = BTreeSet::from_iter(words);

    let len = words.len();
    // 0 => UNK, 1 => PAD
    let word_index = BTreeMap::from_iter(words.into_iter().zip(2..(len + 2)));
    println!("Found {} words", word_index.len());

    let train_reviews = encode_sentences(&train_reviews, &word_index);
    let train_labels: Vec<_> = encode_labels(train_labels);

    let test_reviews = encode_sentences(&test_reviews, &word_index);
    let test_labels: Vec<_> = encode_labels(test_labels);

    let embeddings = net_with_embedding_layer(
        (&train_reviews, &train_labels),
        (&test_reviews, &test_labels),
        len + 2,
    );

    show_similar_embeddings("beautiful", &word_index, &embeddings);
    show_similar_embeddings("terrible", &word_index, &embeddings);

    let embeddings = filling_in_the_blank(&train_reviews, &word_index);

    show_similar_embeddings("beautiful", &word_index, &embeddings);
    show_similar_embeddings("terrible", &word_index, &embeddings);

    analogies(["terrible", "good"], "bad", &word_index, &embeddings);
    analogies(["elizabeth", "he"], "she", &word_index, &embeddings);

    Ok(())
}

fn encode_sentences(v: &[String], word_index: &BTreeMap<&str, usize>) -> Vec<Vec<usize>> {
    v.into_iter()
        .map(|s| {
            let mut encoding = Vec::new();;

            for word in s.split_whitespace() {
                if word_index.contains_key(word) {
                    encoding.push(word_index[word]);
                } else {
                    encoding.push(0);
                }
            }

            encoding
        })
        .collect()
}

fn encode_labels(labels: Vec<u8>) -> Vec<f64> {
    labels
        .into_iter()
        .map(|l| if l > 5 { 1.0 } else { 0.0 })
        .collect()
}

fn net_with_embedding_layer(
    (train_reviews, train_labels): (&[Vec<usize>], &[f64]),
    (test_reviews, test_labels): (&[Vec<usize>], &[f64]),
    vocab_size: usize,
) -> Matrix<f64> {
    let hidden_size = 100;

    let mut embeddings = Matrix::new(
        vocab_size,
        hidden_size,
        generate_random_vector(vocab_size * hidden_size, 0.2, -0.1, &Standard),
    );

    let mut weights_1_2 = Matrix::new(
        hidden_size,
        1,
        generate_random_vector(hidden_size, 0.2, -0.1, &Standard),
    );

    let alpha = 0.01;

    let iterations = 15;

    for _ in 0..iterations {
        let mut train_accuracy = 0.0;
        let mut total = 0.0;

        let progress = ProgressBar::new(train_reviews.len() as u64);
        progress.set_style(
            ProgressStyle::default_bar()
                .template("{msg} {bar:40.cyan/blue} {pos:>7}/{len:7} [{elapsed_precise}]"),
        );

        for (review, label) in train_reviews.iter().zip(train_labels.iter()) {
            // take embeddings
            let mut hidden_layer = Matrix::new(1, hidden_size, vec![0.0; hidden_size]);
            for ix in review.iter() {
                for j in 0..hidden_size {
                    hidden_layer[[0, j]] += embeddings[[*ix, j]];
                }
            }
            sigmoid_mut(&mut hidden_layer);

            let mut prediction = (&hidden_layer).mul(&weights_1_2);
            sigmoid_mut(&mut prediction);

            let delta_2_1 = Matrix::new(1, 1, vec![prediction[[0, 0]] - label]);
            let delta_1_0 = (&delta_2_1).mul(weights_1_2.transpose());

            if prediction[[0, 0]].round() == *label {
                train_accuracy += 1.0;
            }

            total += 1.0;

            let weight_deltas_1_2 = hidden_layer.transpose().mul(delta_2_1);

            for i in 0..hidden_size {
                weights_1_2[[i, 0]] -= alpha * weight_deltas_1_2[[i, 0]];
            }

            for ix in review.iter() {
                for j in 0..hidden_size {
                    embeddings[[*ix, j]] -= alpha * delta_1_0[[0, j]];
                }
            }

            progress.inc(1);
            progress.set_message(&format!("Train Accuracy: {}", train_accuracy / total));
        }

        progress.finish();
    }

    println!("\nEvaluating on Test Dataset\n");

    let progress = ProgressBar::new(test_reviews.len() as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("{msg} {bar:40.cyan/blue} {pos:>7}/{len:7} [{elapsed_precise}]"),
    );

    let mut test_accuracy = 0.0;
    let mut total = 0.0;

    for (review, label) in test_reviews.iter().zip(test_labels.iter()) {
        // take embeddings
        let mut hidden_layer = Matrix::new(1, hidden_size, vec![0.0; hidden_size]);
        for ix in review.iter() {
            for j in 0..hidden_size {
                hidden_layer[[0, j]] += embeddings[[*ix, j]];
            }
        }
        sigmoid_mut(&mut hidden_layer);

        let mut prediction = (&hidden_layer).mul(&weights_1_2);
        sigmoid_mut(&mut prediction);

        if prediction[[0, 0]].round() == *label {
            test_accuracy += 1.0;
        }

        total += 1.0;

        progress.inc(1);
        progress.set_message(&format!("Test Accuracy: {}", test_accuracy / total));
    }

    progress.finish();

    embeddings
}

fn show_similar_embeddings(
    word: &str,
    word_index: &BTreeMap<&str, usize>,
    embeddings: &Matrix<f64>,
) {
    if !word_index.contains_key(word) {
        println!("index does not have {}", word);
    } else {
        let ix = word_index[word];
        let word_embeddings = embeddings.row(ix);

        let sims = get_similar_embeddings(word_embeddings.raw_slice(), word_index, embeddings);

        println!("\nWords Similar to {}:\n", word);
        for i in sims.iter().take(10) {
            println!("{}: {}", i.0, i.1);
        }
    }
}

fn get_similar_embeddings<'a>(
    row: &[f64],
    word_index: &'a BTreeMap<&str, usize>,
    embeddings: &'a Matrix<f64>,
) -> Vec<(&'a str, f64)> {
    let mut sims = Vec::with_capacity(word_index.len());

    for (word, ix) in word_index.into_iter() {
        let mut distance = 0.0;

        for (a, b) in row.iter().zip(embeddings.row(*ix).iter()) {
            distance += (a - b).powi(2);
        }

        sims.push((word.to_owned(), distance.sqrt()));
    }

    sims.sort_by(|a: &(&str, f64), b: &(&str, f64)| {
        if a.1 < b.1 {
            Ordering::Less
        } else if a.1 > b.1 {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    });

    sims
}

fn filling_in_the_blank(
    train_reviews: &[Vec<usize>],
    word_index: &BTreeMap<&str, usize>,
) -> Matrix<f64> {
    let concatenated: Vec<usize> = train_reviews.iter().flat_map(|v| v).cloned().collect();

    // NOTE: inputs are already shuffled

    let hidden_size = 50;
    let (negative_samples, window_size) = (5, 2);
    let alpha = 0.05;

    let iterations = 2;

    let mut weights_0_1 = Matrix::new(
        word_index.len() + 2,
        hidden_size,
        generate_random_vector((word_index.len() + 2) * hidden_size, 0.2, -0.1, &Standard),
    );

    let mut weights_1_2: Matrix<f64> = Matrix::zeros(word_index.len() + 2, hidden_size);

    let mut outputs = Matrix::new(1, negative_samples + 1, vec![0.0; negative_samples + 1]);
    outputs[[0, 0]] = 1.0;

    for _ in 0..iterations {
        let progress = ProgressBar::new(train_reviews.len() as u64);
        progress.set_style(
            ProgressStyle::default_bar()
                .template("{bar:40.cyan/blue} {pos:>7}/{len:7} [{elapsed_precise}]"),
        );

        for review in train_reviews.iter() {
            for target_ix in 0..review.len() {
                let mut target_samples = vec![review[target_ix]];
                target_samples.append(
                    &mut generate_random_vector(negative_samples, 1.0, 0.0, &Standard)
                        .into_iter()
                        .map(|x| (x * (concatenated.len() as f64)) as usize)
                        .map(|ix| concatenated[ix])
                        .collect(),
                );

                let left_window_start =
                    max(0, (target_ix as isize) - (window_size as isize)) as usize;
                let right_window_end = min(target_ix + window_size, review.len());

                let left_window: Vec<usize> = (left_window_start..target_ix)
                    .map(|ix| review[ix])
                    .collect();
                let right_window: Vec<usize> = ((target_ix + 1)..right_window_end)
                    .map(|ix| review[ix])
                    .collect();

                let total_window_size = left_window.len() + right_window.len();

                let mut hidden_layer: Matrix<f64> = Matrix::zeros(1, hidden_size);

                for ix in left_window.iter().chain(right_window.iter()) {
                    for (i, x) in weights_0_1.row(*ix).iter().enumerate() {
                        hidden_layer[[0, i]] += x;
                    }
                }

                for i in 0..total_window_size {
                    hidden_layer[[0, i]] /= total_window_size as f64;
                }

                let mut predictions =
                    (&hidden_layer).mul(select_rows(&weights_1_2, &target_samples).transpose());
                sigmoid_mut(&mut predictions);

                // [1, target_size]
                let layer_2_delta = predictions - (&outputs);

                // [1, hidden_size]
                let layer_1_delta =
                    (&layer_2_delta).mul(select_rows(&weights_1_2, &target_samples));

                // [target_size, hidden_size]
                // NOTE: we have initialized weights_1_2 in reverse order of traditional init
                // normally we'd do hidden_layer.transpose().mul(layer_2_delta)
                let weight_delta_1_2 = layer_2_delta.transpose().mul(hidden_layer);

                for ix in target_samples.into_iter() {
                    for v in 0..hidden_size {
                        weights_1_2[[ix, v]] -= alpha * weight_delta_1_2[[0, v]];
                    }
                }

                for ix in left_window.into_iter().chain(right_window.into_iter()) {
                    for v in 0..hidden_size {
                        weights_0_1[[ix, v]] -= alpha * layer_1_delta[[0, v]];
                    }
                }
            }

            progress.inc(1);
        }

        progress.finish();
    }

    weights_0_1
}

fn select_rows(m: &Matrix<f64>, rows: &[usize]) -> Matrix<f64> {
    Matrix::new(
        rows.len(),
        m.cols(),
        rows.iter().fold(Vec::new(), |mut acc, i| {
            acc.append(&mut Vec::from(m.row(*i).raw_slice()));
            acc
        }),
    )
}

fn analogies(
    positive: [&str; 2],
    negative: &str,
    word_index: &BTreeMap<&str, usize>,
    embeddings: &Matrix<f64>,
) {
    if !word_index.contains_key(positive[0])
        || !word_index.contains_key(positive[1])
        || !word_index.contains_key(negative)
    {
        println!("did not find all words in index");
        return;
    }

    let (pix1, pix2) = (word_index[positive[0]], word_index[positive[1]]);
    let nix = word_index[negative];

    let mut target_row = vec![0.0; embeddings.cols()];
    for i in 0..embeddings.cols() {
        target_row[i] += embeddings[[pix1, i]];
        target_row[i] -= embeddings[[nix, i]];
        target_row[i] += embeddings[[pix2, i]];
    }

    let sims = get_similar_embeddings(&target_row, word_index, embeddings);

    println!("\n{} - {} + {}:\n", positive[0], negative, positive[1]);
    for i in sims.iter().take(10) {
        println!("{}: {}", i.0, i.1);
    }
}
