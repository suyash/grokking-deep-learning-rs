//! Chapter 14 - Learning to Write Like Shakespeare: Long-Short Term Memory
//!
//! https://github.com/iamtrask/Grokking-Deep-Learning/blob/master/Chapter14%20-%20Exploding%20Gradients%20Examples.ipynb
//! https://github.com/iamtrask/Grokking-Deep-Learning/blob/master/Chapter14%20-%20Intro%20to%20LSTMs%20-%20Learn%20to%20Write%20Like%20Shakespeare.ipynb
//! https://github.com/iamtrask/Grokking-Deep-Learning/blob/master/Chapter14%20-%20Intro%20to%20LSTMs%20-%20Part%202%20-%20Learn%20to%20Write%20Like%20Shakespeare.ipynb

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::iter::FromIterator;
use std::ops::Mul;

use datasets::text::shakespeare_100000;
use indicatif::{ProgressBar, ProgressStyle};
use rulinalg::matrix::{BaseMatrix, Matrix};

use grokking_deep_learning_rs::activations::Sigmoid;
use grokking_deep_learning_rs::layers::{Embedding, LSTMCell, Layer, RNNCell};
use grokking_deep_learning_rs::losses::{CrossEntropyLoss, Loss};
use grokking_deep_learning_rs::optimizers::{Optimizer, SGDOptimizer};
use grokking_deep_learning_rs::tensor::Tensor;

fn main() -> Result<(), Box<dyn Error>> {
    println!("\nTraining Shakespeare using RNN Cells\n");
    shakespeare_rnn_cell()?;

    println!("\nVanishing and Exploding Gradients\n");
    vanishing_and_exploding_gradients();

    println!("\nTraining Shakespeare using LSTM Cells\n");
    shakespeare_lstm_cell()?;

    Ok(())
}

fn shakespeare_rnn_cell() -> Result<(), Box<dyn Error>> {
    let embedding_size = 64;
    let rnn_state_size = 512;
    let alpha = 0.05;
    let batch_size = 16;
    let bptt = 25;

    let n_iterations = 1;

    let data = shakespeare_100000()?;

    let characters = BTreeSet::from_iter(data.chars());
    let len = characters.len();
    let word_index = BTreeMap::from_iter(characters.iter().zip(0..len));

    let indices: Vec<_> = data.chars().map(|c| word_index[&c]).collect();

    let embed = Embedding::new(len, embedding_size);
    let cell = RNNCell::new(embedding_size, rnn_state_size, len, Box::new(Sigmoid));

    let criterion = CrossEntropyLoss;

    let mut params = embed.parameters();
    params.append(&mut cell.parameters());

    let optimizer = SGDOptimizer::new(params, alpha);

    let n_batches = (indices.len() as f64 / batch_size as f64).floor() as usize;

    let mut batched_data = Matrix::zeros(n_batches, batch_size);
    for (i, c) in indices.into_iter().enumerate() {
        if i >= batched_data.data().len() {
            break;
        }

        let row = i / n_batches;
        let col = i % n_batches;

        batched_data[[col, row]] = c as f64;
    }

    dbg!(n_batches);

    let n_batches = 100 + 1;

    let steps = (n_batches - 1) / bptt;

    for _ in 0..n_iterations {
        let progress = ProgressBar::new((n_batches - 1) as u64);
        progress.set_style(
            ProgressStyle::default_bar()
                .template("{msg} {bar:40.cyan/blue} {pos:>7}/{len:7} [{elapsed_precise}]"),
        );

        for j in 0..steps {
            let start = bptt * j;

            let mut state = cell.create_start_state(batch_size);

            let mut loss = None;

            for k in 0..bptt {
                let input = batched_data.row(start + k).raw_slice();
                let target = batched_data.row(start + k + 1).raw_slice();

                let input = Tensor::new_const(Matrix::new(1, batch_size, Vec::from(input)));
                let target = Tensor::new_const(Matrix::new(batch_size, 1, Vec::from(target)));

                let rnn_input = &embed.forward(&[&input])[0];
                let mut outputs = cell.forward(&[rnn_input, &state]);

                let output = outputs.remove(0);
                state = outputs.remove(0);

                let current_loss = criterion.forward(&output, &target);
                progress.set_message(&format!(
                    "Batch Loss: {:?}",
                    current_loss.0.borrow().data.data()
                ));

                loss = match loss.take() {
                    None => Some(current_loss),
                    Some(existing_loss) => Some(&existing_loss + &current_loss),
                };

                progress.inc(1);
            }

            loss.unwrap().backward(Tensor::grad(Matrix::ones(1, 1)));
            optimizer.step(true);
        }

        progress.finish();
    }

    Ok(())
}

fn vanishing_and_exploding_gradients() {
    let weights = Matrix::new(2, 2, vec![1.0, 4.0, 4.0, 1.0]);
    let mut activation = sigmoid(Matrix::new(1, 2, vec![1.0, 0.01]));

    println!("Sigmoid Activations");
    let mut activations = Vec::new();
    for _ in 0..10 {
        activation = sigmoid(activation.mul(&weights));
        activations.push(activation.clone());
        println!("{}", activation);
    }

    println!("\nSigmoid Gradients");
    let mut gradient = Matrix::ones(1, 2);
    for activation in activations.into_iter().rev() {
        gradient = activation
            .elemul(&(Matrix::ones(1, 2) - &activation))
            .elemul(&gradient);
        gradient = gradient.mul(weights.transpose());
        println!("{}", gradient);
    }

    println!("\nrelu Activations");
    let mut activations = Vec::new();
    for _ in 0..10 {
        activation = relu(activation.mul(&weights));
        activations.push(activation.clone());
        println!("{}", activation);
    }

    println!("\nrelu Gradients");
    let mut gradient = Matrix::ones(1, 2);
    for activation in activations.into_iter().rev() {
        gradient = gradient.elemul(&Matrix::new(
            1,
            2,
            activation
                .data()
                .iter()
                .map(|v| if v > &0.0 { *v } else { 0.0 })
                .collect::<Vec<f64>>(),
        ));
        gradient = gradient.mul(weights.transpose());
        println!("{}", gradient);
    }
}

fn sigmoid(mut m: Matrix<f64>) -> Matrix<f64> {
    for i in 0..m.rows() {
        for j in 0..m.cols() {
            m[[i, j]] = 1.0 / (1.0 + (-m[[i, j]]).exp());
        }
    }

    m
}

fn relu(mut m: Matrix<f64>) -> Matrix<f64> {
    for i in 0..m.rows() {
        for j in 0..m.cols() {
            m[[i, j]] = if m[[i, j]] > 0.0 { m[[i, j]] } else { 0.0 };
        }
    }

    m
}

fn shakespeare_lstm_cell() -> Result<(), Box<dyn Error>> {
    let embedding_size = 64;
    let rnn_state_size = 512;
    let alpha = 0.05;
    let batch_size = 16;
    let bptt = 25;

    let n_iterations = 1;

    let data = shakespeare_100000()?;

    let characters = BTreeSet::from_iter(data.chars());
    let len = characters.len();
    let word_index = BTreeMap::from_iter(characters.iter().zip(0..len));

    let indices: Vec<_> = data.chars().map(|c| word_index[&c]).collect();

    let embed = Embedding::new(len, embedding_size);
    let cell = LSTMCell::new(embedding_size, rnn_state_size, len);

    let criterion = CrossEntropyLoss;

    let optimizer = SGDOptimizer::new(
        embed
            .parameters()
            .into_iter()
            .chain(cell.parameters().into_iter())
            .collect(),
        alpha,
    );

    let n_batches = (indices.len() as f64 / batch_size as f64).floor() as usize;

    let mut batched_data = Matrix::zeros(n_batches, batch_size);
    for (i, c) in indices.into_iter().enumerate() {
        if i >= batched_data.data().len() {
            break;
        }

        let row = i / n_batches;
        let col = i % n_batches;

        batched_data[[col, row]] = c as f64;
    }

    dbg!(n_batches);

    let n_batches = 100 + 1;

    let steps = (n_batches - 1) / bptt;

    for _ in 0..n_iterations {
        let progress = ProgressBar::new((n_batches - 1) as u64);
        progress.set_style(
            ProgressStyle::default_bar()
                .template("{msg} {bar:40.cyan/blue} {pos:>7}/{len:7} [{elapsed_precise}]"),
        );

        for j in 0..steps {
            let start = bptt * j;

            let (mut h, mut c) = cell.create_start_state(batch_size);

            let mut loss = None;

            for k in 0..bptt {
                let input = batched_data.row(start + k).raw_slice();
                let target = batched_data.row(start + k + 1).raw_slice();

                let input = Tensor::new_const(Matrix::new(1, batch_size, Vec::from(input)));
                let target = Tensor::new_const(Matrix::new(batch_size, 1, Vec::from(target)));

                let rnn_input = &embed.forward(&[&input])[0];
                let mut outputs = cell.forward(&[rnn_input, &h, &c]);

                let output = outputs.remove(0);
                h = outputs.remove(0);
                c = outputs.remove(0);

                let current_loss = criterion.forward(&output, &target);
                progress.set_message(&format!(
                    "Batch Loss: {:?}",
                    current_loss.0.borrow().data.data()
                ));

                loss = match loss.take() {
                    None => Some(current_loss),
                    Some(existing_loss) => Some(&existing_loss + &current_loss),
                };

                progress.inc(1);
            }

            loss.unwrap().backward(Tensor::grad(Matrix::ones(1, 1)));
            optimizer.step(true);
        }

        progress.finish();
    }

    Ok(())
}
