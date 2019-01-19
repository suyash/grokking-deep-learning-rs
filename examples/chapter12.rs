//! Chapter 12 - Introduction to Recurrence - Predicting the Next Word
//! https://github.com/iamtrask/Grokking-Deep-Learning/blob/master/Chapter12%20-%20Intro%20to%20Recurrence%20-%20Predicting%20the%20Next%20Word.ipynb
//!
//! This is *significantly* different from the python version.
//!
//! 1. The dataset is cleaned to remove all whitespaces including tabs, and as a result contains only 19 words. This results in a lower perplexity than
//! the python version from the beginning.
//!
//! 2. The Forward Propagation, Back Propagation and Weight Update steps are implemented in a single function.
//!
//! 3. The gradients explode more rapidly, because of extremely low embeddings to match. Alleviated this by lowering the alpha from 0.001 to 0.0005
//! and increasing embedding size from 10 to 100. Another measure would be to cap the gradients.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::iter::FromIterator;
use std::ops::Mul;

use datasets::text::babi_en_single_supporting_fact_task;
use indicatif::{ProgressBar, ProgressStyle};
use rand::distributions::Uniform;
use rulinalg::matrix::{BaseMatrix, Matrix};

use grokking_deep_learning_rs::{generate_random_vector, softmax_mut, argmax};

fn main() -> Result<(), Box<dyn Error>> {
    embeddings_forward_propagation();

    let (train_data, _) = babi_en_single_supporting_fact_task()?;

    let train_data: Vec<Vec<String>> = train_data
        .map(|v| vec![v.0, v.1, (v.2).0])
        .flat_map(|v| v.into_iter())
        .map(|s| {
            s.split_whitespace()
                .map(|w| {
                    w.chars()
                        .filter(|c| (*c >= 'a' && *c <= 'z') || (*c >= 'A' && *c <= 'Z'))
                        .collect()
                })
                .collect()
        })
        .collect();

    let words = BTreeSet::from_iter(train_data.iter().flat_map(|v| v.iter()));

    let word_count = words.len();
    let word_index = BTreeMap::from_iter(words.into_iter().zip(0..word_count));
    let inverted_word_index = BTreeMap::from_iter(word_index.clone().into_iter().map(|(k, v)| (v, k)));

    let (start_state, embeddings, recurrent_weights, state_to_prediction_weights) =
        training_with_arbitrary_length(&train_data, &word_index)?;

    let sentence = &train_data[0];

    let mut current_state = start_state.clone();

    for (i, word) in sentence.iter().take(sentence.len() - 1).enumerate() {
        let mut prediction = (&current_state).mul(&state_to_prediction_weights);
        softmax_mut(&mut prediction);

        let pred_ix = argmax(prediction.row(0).raw_slice());
        let predicted_word = inverted_word_index[&pred_ix];

        println!("Input: {}, Expected: {}, Predicted: {}", word, sentence[i + 1], predicted_word);
        current_state = current_state.mul(&recurrent_weights) + embeddings.row(word_index[word]).into_matrix();
    }

    Ok(())
}

fn embeddings_forward_propagation() {
    let mut word_vectors = BTreeMap::new();
    word_vectors.insert("yankees", Matrix::new(1, 3, vec![0.0; 3]));
    word_vectors.insert("bears", Matrix::new(1, 3, vec![0.0; 3]));
    word_vectors.insert("braves", Matrix::new(1, 3, vec![0.0; 3]));
    word_vectors.insert("red", Matrix::new(1, 3, vec![0.0; 3]));
    word_vectors.insert("socks", Matrix::new(1, 3, vec![0.0; 3]));
    word_vectors.insert("lose", Matrix::new(1, 3, vec![0.0; 3]));
    word_vectors.insert("defeat", Matrix::new(1, 3, vec![0.0; 3]));
    word_vectors.insert("beat", Matrix::new(1, 3, vec![0.0; 3]));
    word_vectors.insert("tie", Matrix::new(1, 3, vec![0.0; 3]));

    let sent_to_output_weights =
        Matrix::new(3, word_vectors.len(), vec![0.0; 3 * word_vectors.len()]);

    let weights: Matrix<f64> = Matrix::identity(3);

    let layer_0 = &word_vectors["red"];
    let layer_1 = layer_0.mul(&weights) + &word_vectors["socks"];
    let layer_2 = layer_1.mul(&weights) + &word_vectors["defeat"];

    let mut prediction = layer_2.mul(&sent_to_output_weights);
    softmax_mut(&mut prediction);

    println!("{}", prediction);
}

#[allow(clippy::type_complexity)]
fn training_with_arbitrary_length(
    train_data: &[Vec<String>],
    word_index: &BTreeMap<&String, usize>,
) -> Result<(Matrix<f64>, Matrix<f64>, Matrix<f64>, Matrix<f64>), Box<dyn Error>> {
    let word_count = word_index.len();

    let embedding_size = 50;

    let distribution = Uniform::new(0.0, 1.0);

    let mut embeddings = Matrix::new(
        word_count,
        embedding_size,
        generate_random_vector(word_count * embedding_size, 0.1, -0.05, &distribution),
    );

    let mut recurrent_weights = Matrix::identity(embedding_size);

    let mut state_to_prediction_weights = Matrix::new(
        embedding_size,
        word_count,
        generate_random_vector(embedding_size * word_count, 0.1, -0.05, &distribution),
    );

    let word_target_embeddings = Matrix::identity(word_count);

    let mut start_state = Matrix::zeros(1, embedding_size);

    let alpha = 0.0004;

    for _ in 0..10 {
        let progress = ProgressBar::new(train_data.len() as u64);
        progress.set_style(
            ProgressStyle::default_bar()
                .template("{msg} {bar:40.cyan/blue} {pos:>7}/{len:7} [{elapsed_precise}]"),
        );

        for sentence in train_data.iter() {
            // forward prop

            let mut current_state = start_state.clone();
            let mut loss = 0.0;

            let mut cells = Vec::with_capacity(sentence.len());
            cells.push((None, current_state.clone()));

            for word in sentence.iter().skip(1) {
                let mut prediction = (&current_state).mul(&state_to_prediction_weights);
                softmax_mut(&mut prediction);

                loss += -(prediction[[0, word_index[word]]]).ln();

                let mut next_state = (&current_state).mul(&recurrent_weights);

                for i in 0..embedding_size {
                    next_state[[0, i]] += embeddings[[word_index[word], i]];
                }

                cells.push((Some(prediction), next_state.clone()));

                current_state = next_state;
            }

            loss /= (sentence.len() - 1) as f64;

            // backward prop

            let mut deltas: Vec<(Option<Matrix<f64>>, Matrix<f64>)> = Vec::new();

            let mut current_state_delta: Matrix<f64> = Matrix::identity(1);

            for (i, (prediction, _)) in cells.iter().enumerate().rev() {
                let prediction_delta = match prediction {
                    Some(prediction) => Some(
                        prediction
                            - (word_target_embeddings
                                .row(word_index[&sentence[i]])
                                .into_matrix()),
                    ),
                    None => None,
                };

                let mut state_delta_from_predictions = match &prediction_delta {
                    Some(prediction_delta) => {
                        Some(prediction_delta.mul(state_to_prediction_weights.transpose()))
                    }
                    None => None,
                };

                let mut state_delta_from_next_state = if i == cells.len() - 1 {
                    None
                } else {
                    Some(current_state_delta.mul(recurrent_weights.transpose()))
                };

                current_state_delta = match (
                    state_delta_from_predictions.take(),
                    state_delta_from_next_state.take(),
                ) {
                    (Some(m1), Some(m2)) => m1 + m2,
                    (Some(m1), None) => m1,
                    (None, Some(m2)) => m2,
                    _ => panic!("this is broken"),
                };

                deltas.push((prediction_delta, current_state_delta.clone()));
            }

            // weights update

            // align deltas with cells
            deltas.reverse();

            let (_, start_delta) = &deltas[0];
            for i in 0..embedding_size {
                start_state[[0, i]] -=
                    (alpha * start_delta[[0, i]]) / ((sentence.len() - 1) as f64);
            }

            for i in 1..cells.len() {
                let (_, state) = &cells[i];
                let (prediction_delta, state_delta) = &deltas[i];
                // let (_, prev_state) = &cells[i - 1];

                let prediction_delta = prediction_delta.as_ref().unwrap();

                let state_to_prediction_weights_delta = state.transpose().mul(prediction_delta);
                for j in 0..embedding_size {
                    for k in 0..word_count {
                        state_to_prediction_weights[[j, k]] -= (alpha
                            * state_to_prediction_weights_delta[[j, k]])
                            / ((sentence.len() - 1) as f64);
                    }
                }

                for j in 0..embedding_size {
                    embeddings[[word_index[&sentence[i]], j]] -=
                        (alpha * state_delta[[0, j]]) / ((sentence.len() - 1) as f64);
                }

                let recurrent_weights_delta = state.transpose().mul(state_delta);
                for j in 0..embedding_size {
                    for k in 0..embedding_size {
                        recurrent_weights[[j, k]] -= (alpha * recurrent_weights_delta[[j, k]])
                            / ((sentence.len() - 1) as f64);
                    }
                }
            }

            progress.set_message(&format!("Perplexity: {}", loss.exp()));
            progress.inc(1);
        }

        progress.finish();
    }

    Ok((
        start_state,
        embeddings,
        recurrent_weights,
        state_to_prediction_weights,
    ))
}
