//! Chapter 15: Introduction to Federated Learning
//!
//! https://github.com/iamtrask/Grokking-Deep-Learning/blob/master/Chapter15%20-%20Intro%20to%20Federated%20Learning%20-%20Deep%20Learning%20on%20Unseen%20Data.ipynb

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::iter::FromIterator;

use datasets::text::enron_spam;
use datasets::Dataset;
use indicatif::{ProgressBar, ProgressStyle};
// use paillier::traits::{Add, Decrypt, Encrypt, KeyGeneration, Mul};
// use paillier::{EncryptionKey, Paillier};
use rulinalg::matrix::Matrix;

use grokking_deep_learning_rs::layers::{Embedding, Layer};
use grokking_deep_learning_rs::losses::{Loss, MSELoss};
use grokking_deep_learning_rs::optimizers::{Optimizer, SGDOptimizer};
use grokking_deep_learning_rs::tensor::{Sum, Tensor};

fn main() -> Result<(), Box<dyn Error>> {
    // println!("\nRegular Deep Learning\n");
    // regular_deep_learning()?;

    println!("\nFederated Deep Learning\n");
    federated_deep_learning()?;

    Ok(())
}

/// Regular Deep Learning

fn regular_deep_learning() -> Result<(), Box<dyn Error>> {
    let (spam, ham) = enron_spam()?;

    let dataset_size = 3000;
    let max_sentence_len = 100;

    let (spam, ham) = (
        parse_dataset(spam, dataset_size, max_sentence_len),
        parse_dataset(ham, dataset_size, max_sentence_len),
    );

    let word_index = {
        let words = BTreeSet::from_iter(spam.iter().chain(ham.iter()).flat_map(|v| v.iter()));
        let word_count = words.len();
        BTreeMap::from_iter(words.into_iter().zip(0..word_count))
    };

    let word_count = word_index.len();

    dbg!(word_count);

    let spam = spam
        .iter()
        .map(|sentence| {
            sentence
                .into_iter()
                .map(|word| word_index[&word] as f64)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let ham = ham
        .iter()
        .map(|sentence| {
            sentence
                .into_iter()
                .map(|word| word_index[&word] as f64)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let train_data = spam
        .iter()
        .take(dataset_size / 2)
        .cloned()
        .zip(vec![1.0; dataset_size / 2])
        .chain(
            ham.iter()
                .take(dataset_size / 2)
                .cloned()
                .zip(vec![0.0; dataset_size / 2]),
        )
        .shuffle(dataset_size, 0)
        .collect::<Vec<_>>();

    let test_data = spam
        .iter()
        .skip(dataset_size / 2)
        .cloned()
        .zip(vec![1.0; dataset_size / 2])
        .chain(
            ham.iter()
                .skip(dataset_size / 2)
                .cloned()
                .zip(vec![0.0; dataset_size / 2]),
        )
        .shuffle(dataset_size, 0)
        .collect::<Vec<_>>();

    let model = Embedding::new(word_count, 1);

    {
        model.weights.0.borrow_mut().data *= 0.0;
    }

    let n_iterations = 10;
    let batch_size = 200;
    let n_batches = dataset_size / batch_size;

    let model = train(
        model,
        train_data,
        dataset_size,
        &word_index,
        max_sentence_len,
        n_iterations,
        n_batches,
        batch_size,
    );

    let accuracy = test(&model, &test_data, dataset_size, max_sentence_len);

    println!("Test Accuracy: {}", accuracy);

    Ok(())
}

fn train(
    model: Embedding,
    data: Vec<(Vec<f64>, f64)>,
    dataset_size: usize,
    word_index: &BTreeMap<&String, usize>,
    max_sentence_len: usize,
    n_iterations: usize,
    n_batches: usize,
    batch_size: usize,
) -> Embedding {
    // NOTE: Unlike the Python version, cannot do batching as cannot support 3D operations
    // so running stochastic gradient descent in batch_size iterations and accumulating loss
    let criterion = MSELoss;
    let optim = SGDOptimizer::new(model.parameters(), 0.01);

    for _ in 0..n_iterations {
        let progress = ProgressBar::new(n_batches as u64);
        progress.set_style(
            ProgressStyle::default_bar()
                .template("{msg} {bar:40.cyan/blue} {pos:>7}/{len:7} [{elapsed_precise}]"),
        );

        let mut total_loss = 0.0;

        for bi in 0..n_batches {
            let mut current_loss = 0.0;

            {
                model.weights.0.borrow_mut().data[[word_index[&"<UNK>".to_owned()], 0]] *= 0.0;
            }

            for i in (batch_size * bi)..(batch_size * (bi + 1)) {
                let (input, output) = &data[i];
                let input = Tensor::new_const(Matrix::new(1, max_sentence_len, input.clone()));
                let prediction = model.forward(&[&input]).remove(0);
                let prediction = prediction.sum(0);
                let prediction = prediction.sigmoid();

                let target = Tensor::new_const(Matrix::new(1, 1, vec![*output]));

                let loss = criterion.forward(&prediction, &target);

                current_loss += loss.0.borrow().data.data()[0];

                loss.backward(Tensor::grad(Matrix::ones(1, 1)));
                optim.step(true);
            }

            total_loss += current_loss;

            progress.set_message(&format!("Loss: {}", current_loss / (batch_size as f64)));
            progress.inc(1);
        }

        progress.finish_with_message(&format!("Loss: {}", total_loss / (dataset_size as f64)));
    }

    model
}

fn test(
    model: &Embedding,
    data: &Vec<(Vec<f64>, f64)>,
    dataset_size: usize,
    max_sentence_len: usize,
) -> f64 {
    let mut accuracy = 0.0;

    for i in 0..(dataset_size / 2) {
        let (input, output) = &data[i];
        let input = Tensor::new_const(Matrix::new(1, max_sentence_len, input.clone()));
        let prediction = model.forward(&[&input]).remove(0);
        let prediction = prediction.sum(0);
        let prediction = prediction.sigmoid();

        if (prediction.0.borrow().data.data()[0] >= 0.5 && output == &1.0)
            || (prediction.0.borrow().data.data()[0] < 0.5 && output == &0.0)
        {
            accuracy += 1.0;
        }
    }

    accuracy / ((dataset_size / 2) as f64)
}

fn parse_dataset(
    dataset: impl Dataset<Item = String>,
    dataset_size: usize,
    max_sentence_len: usize,
) -> Vec<Vec<String>> {
    dataset
        .take(dataset_size)
        .map(|email| {
            email
                .split("\n")
                .map(|line| line.split_whitespace())
                .flat_map(|v| v)
                .map(|v| {
                    v.chars()
                        .filter(|c| (c >= &'a' && c <= &'z') || (c >= &'A' && c <= &'Z'))
                        .collect::<String>()
                })
                .collect::<Vec<_>>()
        })
        .map(|mut email| {
            if email.len() >= max_sentence_len {
                email.drain(max_sentence_len..email.len());
            } else {
                for _ in 0..(max_sentence_len - email.len()) {
                    email.push("<UNK>".to_owned());
                }
            }

            email
        })
        .collect()
}

fn federated_deep_learning() -> Result<(), Box<dyn Error>> {
    let (spam, ham) = enron_spam()?;

    let dataset_size = 4000;
    let train_dataset_size = 3000;
    let test_dataset_size = dataset_size - train_dataset_size;
    let max_sentence_len = 100;

    let (spam, ham) = (
        parse_dataset(spam, dataset_size, max_sentence_len),
        parse_dataset(ham, dataset_size, max_sentence_len),
    );

    let word_index = {
        let words = BTreeSet::from_iter(spam.iter().chain(ham.iter()).flat_map(|v| v.iter()));
        let word_count = words.len();
        BTreeMap::from_iter(words.into_iter().zip(0..word_count))
    };

    let word_count = word_index.len();

    dbg!(word_count);

    let spam = spam
        .iter()
        .map(|sentence| {
            sentence
                .into_iter()
                .map(|word| word_index[&word] as f64)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let ham = ham
        .iter()
        .map(|sentence| {
            sentence
                .into_iter()
                .map(|word| word_index[&word] as f64)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let train_data = spam
        .iter()
        .take(train_dataset_size)
        .cloned()
        .zip(vec![1.0; train_dataset_size])
        .chain(
            ham.iter()
                .take(train_dataset_size)
                .cloned()
                .zip(vec![0.0; train_dataset_size]),
        )
        .shuffle(2 * train_dataset_size, 0)
        .collect::<Vec<_>>();

    let test_data = spam
        .iter()
        .skip(train_dataset_size)
        .cloned()
        .zip(vec![1.0; test_dataset_size])
        .chain(
            ham.iter()
                .skip(train_dataset_size)
                .cloned()
                .zip(vec![0.0; test_dataset_size]),
        )
        .shuffle(2 * test_dataset_size, 0)
        .collect::<Vec<_>>();

    let alice: Vec<_> = train_data
        .iter()
        .take(train_dataset_size / 3)
        .cloned()
        .collect();
    let bob: Vec<_> = train_data
        .iter()
        .skip(train_dataset_size / 3)
        .take(train_dataset_size / 3)
        .cloned()
        .collect();
    let charlie: Vec<_> = train_data
        .iter()
        .skip(2 * train_dataset_size / 3)
        .cloned()
        .collect();

    let alice_model = Embedding::new(word_count, 1);
    let bob_model = Embedding::new(word_count, 1);
    let charlie_model = Embedding::new(word_count, 1);

    {
        alice_model.weights.0.borrow_mut().data *= 0.0;
        bob_model.weights.0.borrow_mut().data *= 0.0;
        charlie_model.weights.0.borrow_mut().data *= 0.0;
    }

    let n_iterations = 10;
    let batch_size = 200;
    let n_batches = train_dataset_size / (3 * batch_size);

    println!("Training Alice");
    let alice_model = train(
        alice_model,
        alice,
        train_dataset_size / 3,
        &word_index,
        max_sentence_len,
        n_iterations,
        n_batches,
        batch_size,
    );

    println!("Training Bob");
    let bob_model = train(
        bob_model,
        bob,
        train_dataset_size / 3,
        &word_index,
        max_sentence_len,
        n_iterations,
        n_batches,
        batch_size,
    );

    println!("Training Charlie");
    let charlie_model = train(
        charlie_model,
        charlie,
        train_dataset_size / 3,
        &word_index,
        max_sentence_len,
        n_iterations,
        n_batches,
        batch_size,
    );

    let alice_weights = &alice_model.weights.0.borrow().data;
    let bob_weights = &bob_model.weights.0.borrow().data;
    let charlie_weights = &charlie_model.weights.0.borrow().data;

    let weights = alice_weights + bob_weights + charlie_weights;
    let weights = weights / 3.0;

    let model = Embedding::from_weights(weights);

    let accuracy = test(&model, &test_data, dataset_size, max_sentence_len);

    println!("Test Accuracy: {}", accuracy);

    Ok(())
}

// fn train_and_encrypt(
//     model: Embedding,
//     data: Vec<(Vec<f64>, f64)>,
//     dataset_size: usize,
//     word_index: &BTreeMap<&String, usize>,
//     max_sentence_len: usize,
//     n_iterations: usize,
//     n_batches: usize,
//     batch_size: usize,
//     encryption_key: &EncryptionKey,
// ) -> Vec<paillier::encoding::EncodedCiphertext<f64>> {
//     let model = train(
//         model,
//         data,
//         dataset_size,
//         word_index,
//         max_sentence_len,
//         n_iterations,
//         n_batches,
//         batch_size,
//     );
//
//     model
//         .weights
//         .0
//         .borrow()
//         .data
//         .data()
//         .iter()
//         .map(|v| Paillier::encrypt(&encryption_key, *v))
//         .collect()
// }
