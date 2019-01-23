//! Chapter13 - Intro to Automatic Differentiation - Let's Build A Deep Learning Framework
//!
//! https://github.com/iamtrask/Grokking-Deep-Learning/blob/master/Chapter13%20-%20Intro%20to%20Automatic%20Differentiation%20-%20Let's%20Build%20A%20Deep%20Learning%20Framework.ipynb

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::iter::FromIterator;
use std::ops::Add;

use datasets::text::babi_en_single_supporting_fact_task;
use datasets::Dataset;
use rand::distributions::Uniform;
use rulinalg::matrix::{BaseMatrix, Matrix};

use grokking_deep_learning_rs::activations::{Sigmoid, Tanh};
use grokking_deep_learning_rs::layers::{Embedding, Layer, Linear, RNNCell, Sequential};
use grokking_deep_learning_rs::losses::{CrossEntropyLoss, Loss, MSELoss};
use grokking_deep_learning_rs::optimizers::{Optimizer, SGDOptimizer};
use grokking_deep_learning_rs::tensor::{Dot, Sum, Tensor};
use grokking_deep_learning_rs::{argmax, generate_random_vector};

fn main() {
    println!("\nIntroduction to Tensors\n");
    introduction_to_tensors();

    println!("\nIntroduction to Autograd\n");
    introduction_to_autograd();
    introduction_to_autograd_2();

    println!("\nAutograd with multiple tensors\n");
    autograd_with_multiple_tensors();
    autograd_neg();

    println!("\nUsing Autograd ot train a Neural Network\n");
    training_using_autograd();

    println!("\nAdding Automatic Optimization\n");
    training_with_automatic_optimization();

    println!("\nLayers Which Contain Layers\n");
    layers_which_contain_layers();

    println!("\nLoss Function Layers\n");
    loss_function_layers();

    println!("\nNonLinearity Layers\n");
    nonlinearity_layers();

    println!("\nEmbedding Layers\n");
    embedding_layer();

    println!("\nThe Embedding Layer\n");
    cross_entropy_loss();

    println!("\nRecurrent Neural Network\n");
    recurrent_neural_network().unwrap();
}

fn introduction_to_tensors() {
    let t1 = BasicTensor1 { data: vec![0.0] };
    let t2 = BasicTensor1 { data: vec![1.0] };
    println!("{:?}", t1 + t2);
}

#[derive(Debug)]
struct BasicTensor1 {
    data: Vec<f64>,
}

impl Add for BasicTensor1 {
    type Output = BasicTensor1;

    fn add(self, other: BasicTensor1) -> Self::Output {
        BasicTensor1 {
            data: self
                .data
                .into_iter()
                .zip(other.data.into_iter())
                .map(|(a, b)| a + b)
                .collect(),
        }
    }
}

fn introduction_to_autograd() {
    let x = BasicTensor2::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = BasicTensor2::new(vec![2.0; 5]);

    let mut z = x + y;
    println!("{:?}", z);

    z.backward(BasicTensor2::new(vec![1.0, 1.0, 1.0, 1.0, 1.0]));

    let xy = z.creators.unwrap();

    println!("{:?}", xy[0].grad);
    println!("{:?}", xy[1].grad);
}

#[derive(Debug, Clone)]
enum BasicOperation {
    Add,
    Const,
}

#[derive(Debug, Clone)]
struct BasicTensor2 {
    data: Vec<f64>,
    grad: Option<Box<BasicTensor2>>,
    creation_op: BasicOperation,
    creators: Option<Vec<BasicTensor2>>,
}

impl BasicTensor2 {
    fn new(data: Vec<f64>) -> Self {
        BasicTensor2 {
            data,
            grad: None,
            creation_op: BasicOperation::Const,
            creators: None,
        }
    }

    fn backward(&mut self, grad: BasicTensor2) {
        match self.creation_op {
            BasicOperation::Add => {
                for c in self.creators.as_mut().unwrap().iter_mut() {
                    c.backward(grad.clone());
                }
            }
            _ => {
                self.grad = Some(Box::new(grad));
            }
        };
    }
}

impl Add for BasicTensor2 {
    type Output = BasicTensor2;

    fn add(self, other: Self) -> BasicTensor2 {
        BasicTensor2 {
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a + b)
                .collect(),
            grad: None,
            creation_op: BasicOperation::Add,
            creators: Some(vec![self, other]),
        }
    }
}

#[allow(clippy::many_single_char_names)]
fn introduction_to_autograd_2() {
    let a = BasicTensor2::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let b = BasicTensor2::new(vec![2.0; 5]);
    let c = BasicTensor2::new(vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    let d = BasicTensor2::new(vec![-1.0, -2.0, -3.0, -4.0, -5.0]);

    let e = a + b;
    let f = c + d;
    let mut g = e + f;

    g.backward(BasicTensor2::new(vec![1.0, 1.0, 1.0, 1.0, 1.0]));
    println!("{:?}", g);

    let ef = g.creators.as_ref().unwrap();
    let ab = ef[0].creators.as_ref().unwrap();

    let a = &ab[0];
    println!("{:?}", a.grad);
}

#[allow(clippy::many_single_char_names)]
fn autograd_with_multiple_tensors() {
    let a = Tensor::new_const(Matrix::new(1, 5, vec![1.0, 2.0, 3.0, 4.0, 5.0]));
    let b = Tensor::new_const(Matrix::new(1, 5, vec![2.0, 2.0, 2.0, 2.0, 2.0]));
    let c = Tensor::new_const(Matrix::new(1, 5, vec![5.0, 4.0, 3.0, 2.0, 1.0]));

    let d = &a + &b;
    let e = &b + &c;
    let f = &d + &e;

    // println!("{:#?}", f);
    f.backward(Tensor::grad(Matrix::new(
        1,
        5,
        vec![1.0, 1.0, 1.0, 1.0, 1.0],
    )));
    println!("{:?}", b.0.borrow().grad);
}

#[allow(clippy::many_single_char_names)]
fn autograd_neg() {
    let a = Tensor::new_const(Matrix::new(1, 5, vec![1.0, 2.0, 3.0, 4.0, 5.0]));
    let b = Tensor::new_const(Matrix::new(1, 5, vec![2.0, 2.0, 2.0, 2.0, 2.0]));
    let c = Tensor::new_const(Matrix::new(1, 5, vec![5.0, 4.0, 3.0, 2.0, 1.0]));

    let d = &a + &(-&b);
    let e = &(-&b) + &c;
    let f = &d + &e;

    f.backward(Tensor::grad(Matrix::new(
        1,
        5,
        vec![1.0, 1.0, 1.0, 1.0, 1.0],
    )));
    println!("{:?}", b.0.borrow().grad);
}

/// Using Autograd to train a Neural Network

fn training_using_autograd() {
    let data = Tensor::new_const(Matrix::new(
        4,
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    ));
    let target = Tensor::new_const(Matrix::new(4, 1, vec![0.0, 1.0, 0.0, 1.0]));

    let distribution = Uniform::new(0.0, 1.0);

    let w1 = Tensor::new_const(Matrix::new(
        2,
        3,
        generate_random_vector(2 * 3, 1.0, 0.0, &distribution),
    ));
    let w2 = Tensor::new_const(Matrix::new(
        3,
        1,
        generate_random_vector(3, 1.0, 0.0, &distribution),
    ));

    let alpha = 0.1;

    for _ in 0..10 {
        let pred = data.dot(&w1).dot(&w2);
        let loss = (&(&pred - &target) * &(&pred - &target)).sum(0);
        let (loss_rows, loss_cols) = (1, 1);

        println!("Loss: {:?}", loss.0.borrow().data);

        loss.backward(Tensor::grad(Matrix::ones(loss_rows, loss_cols)));

        {
            let mut w1 = w1.0.borrow_mut();
            let grad = w1.grad.take();
            w1.grad = None;

            let grad = grad.unwrap();
            let grad = &grad.borrow().data;

            for i in 0..w1.data.rows() {
                for j in 0..w1.data.cols() {
                    w1.data[[i, j]] -= alpha * grad[[i, j]];
                }
            }
        }

        {
            let mut w2 = w2.0.borrow_mut();
            let grad = w2.grad.take();
            w2.grad = None;

            let grad = grad.unwrap();
            let grad = &grad.borrow().data;

            for i in 0..w2.data.rows() {
                for j in 0..w2.data.cols() {
                    w2.data[[i, j]] -= alpha * grad[[i, j]];
                }
            }
        }
    }
}

/// Adding Automatic Optimization

fn training_with_automatic_optimization() {
    let data = Tensor::new_const(Matrix::new(
        4,
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    ));
    let target = Tensor::new_const(Matrix::new(4, 1, vec![0.0, 1.0, 0.0, 1.0]));

    let distribution = Uniform::new(0.0, 1.0);

    let w1 = Tensor::new_const(Matrix::new(
        2,
        3,
        generate_random_vector(2 * 3, 1.0, 0.0, &distribution),
    ));

    let w2 = Tensor::new_const(Matrix::new(
        3,
        1,
        generate_random_vector(3, 1.0, 0.0, &distribution),
    ));

    let alpha = 0.1;

    let optimizer = SGDOptimizer::new(vec![&w1, &w2], alpha);

    for _ in 0..10 {
        // predict
        let pred = data.dot(&w1).dot(&w2);

        // compare
        let loss = (&(&pred - &target) * &(&pred - &target)).sum(0);
        let (loss_rows, loss_cols) = (1, 1);

        println!("Loss: {:?}", loss.0.borrow().data.data());

        // calculate difference
        loss.backward(Tensor::grad(Matrix::ones(loss_rows, loss_cols)));

        // learn
        optimizer.step(true);
    }
}

/// Layers Which Contain Layers

fn layers_which_contain_layers() {
    let data = Tensor::new_const(Matrix::new(
        4,
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    ));

    let target = Tensor::new_const(Matrix::new(4, 1, vec![0.0, 1.0, 0.0, 1.0]));

    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 3)),
        Box::new(Linear::new(3, 1)),
    ]);

    let optim = SGDOptimizer::new(model.parameters(), 0.05);

    for _ in 0..10 {
        let pred = model.forward(&[&data]);

        // compare
        let loss = (&(&pred[0] - &target) * &(&pred[0] - &target)).sum(0);

        println!("Loss: {:?}", loss.0.borrow().data.data());

        // calculate difference
        loss.backward(Tensor::grad(Matrix::ones(1, 1)));

        // learn
        optim.step(true);
    }
}

fn loss_function_layers() {
    let data = Tensor::new_const(Matrix::new(
        4,
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    ));

    let target = Tensor::new_const(Matrix::new(4, 1, vec![0.0, 1.0, 0.0, 1.0]));

    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 3)),
        Box::new(Linear::new(3, 1)),
    ]);

    let criterion = MSELoss;
    let optim = SGDOptimizer::new(model.parameters(), 0.05);

    for _ in 0..10 {
        let pred = model.forward(&[&data]);

        // compare
        let loss = criterion.forward(&pred[0], &target);

        println!("Loss: {:?}", loss.0.borrow().data.data());

        // calculate difference
        loss.backward(Tensor::grad(Matrix::ones(1, 1)));

        // learn
        optim.step(true);
    }
}

/// NonLinearity Layers

fn nonlinearity_layers() {
    let data = Tensor::new_const(Matrix::new(
        4,
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    ));

    let target = Tensor::new_const(Matrix::new(4, 1, vec![0.0, 1.0, 0.0, 1.0]));

    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 3)),
        Box::new(Tanh),
        Box::new(Linear::new(3, 1)),
        Box::new(Sigmoid),
    ]);

    let criterion = MSELoss;
    let optim = SGDOptimizer::new(model.parameters(), 0.5);

    for _ in 0..10 {
        let pred = model.forward(&[&data]);

        // compare
        let loss = criterion.forward(&pred[0], &target);

        println!("Loss: {:?}", loss.0.borrow().data.data());

        // calculate difference
        loss.backward(Tensor::grad(Matrix::ones(1, 1)));

        // learn
        optim.step(true);
    }
}

/// The Embedding Layer

fn embedding_layer() {
    let data = Tensor::new_const(Matrix::new(1, 4, vec![1.0, 2.0, 1.0, 2.0]));
    let target = Tensor::new_const(Matrix::new(4, 1, vec![0.0, 1.0, 0.0, 1.0]));

    let model = Sequential::new(vec![
        Box::new(Embedding::new(5, 3)),
        Box::new(Tanh),
        Box::new(Linear::new(3, 1)),
        Box::new(Sigmoid),
    ]);

    let criterion = MSELoss;
    let optim = SGDOptimizer::new(model.parameters(), 0.07);

    for _ in 0..10 {
        let pred = model.forward(&[&data]);

        // compare
        let loss = criterion.forward(&pred[0], &target);

        println!("Loss: {:?}", loss.0.borrow().data.data());

        // calculate difference
        loss.backward(Tensor::grad(Matrix::ones(1, 1)));

        // learn
        optim.step(true);
    }
}

/// The Cross Entropy Layer

fn cross_entropy_loss() {
    let data = Tensor::new_const(Matrix::new(1, 4, vec![1.0, 2.0, 1.0, 2.0]));
    let target = Tensor::new_const(Matrix::new(4, 1, vec![0.0, 1.0, 0.0, 1.0]));

    let model = Sequential::new(vec![
        Box::new(Embedding::new(3, 3)),
        Box::new(Tanh),
        Box::new(Linear::new(3, 4)),
    ]);

    let criterion = CrossEntropyLoss;
    let optim = SGDOptimizer::new(model.parameters(), 0.1);

    for _ in 0..10 {
        let pred = model.forward(&[&data]);
        // println!("pred {}", pred.0.borrow().data);

        // compare
        let loss = criterion.forward(&pred[0], &target);

        println!("Loss: {:?}", loss.0.borrow().data.data());

        // calculate difference
        loss.backward(Tensor::grad(Matrix::ones(1, 1)));

        // learn
        optim.step(true);
    }
}

#[allow(clippy::needless_range_loop)]
fn recurrent_neural_network() -> Result<(), Box<dyn Error>> {
    let (train_data, _) = babi_en_single_supporting_fact_task()?;

    let train_data: Vec<Vec<String>> = train_data
        .map(|v| vec![v.0, v.1 /*, (v.2).0*/])
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

    let total_data_size = train_data.len();

    let words = BTreeSet::from_iter(train_data.iter().flat_map(|v| v.iter()));

    let word_count = words.len();
    let word_index = BTreeMap::from_iter(words.into_iter().zip(0..word_count));
    let inverted_word_index =
        BTreeMap::from_iter(word_index.clone().into_iter().map(|(k, v)| (v, k)));

    let train_data: Vec<Vec<f64>> = train_data
        .iter()
        .map(|s| s.iter().map(|w| word_index[w] as f64).collect())
        .collect();

    let max_len = train_data.iter().map(|s| s.len()).max().unwrap();
    let pad = word_index.len() + 1;

    let batch_size = 250;

    let train_data: Vec<_> = train_data
        .into_iter()
        .batch(batch_size, true)
        .map(|v: Vec<Vec<f64>>| {
            let mut ans = vec![vec![0.0; batch_size]; max_len];
            for i in 0..batch_size {
                for j in 0..v[i].len() {
                    ans[j][i] = v[i][j];
                }

                for j in v[i].len()..max_len {
                    ans[j][i] = pad as f64;
                }
            }

            ans
        })
        .collect();

    let embedding_size = 16;

    // net
    let embed = Embedding::new(word_index.len() + 2, embedding_size);
    let model = RNNCell::new(embedding_size, 16, word_index.len() + 2, Box::new(Sigmoid));

    let criterion = CrossEntropyLoss;
    let mut parameters = embed.parameters();
    parameters.append(&mut model.parameters());

    let optim = SGDOptimizer::new(parameters, 0.01);

    for _ in 0..10 {
        let mut total_loss = 0.0;
        let mut total_accuracy = 0.0;

        for batch in train_data.iter() {
            let mut hidden = model.create_start_state(batch_size);
            let mut output = None;

            let len = batch.len();

            for row in batch.iter().take(len - 1) {
                let input = Tensor::new_const(Matrix::new(1, batch_size, row.clone()));
                let rnn_input = embed.forward(&[&input]).remove(0);
                let mut outputs = model.forward(&[&rnn_input, &hidden]);
                output = Some(outputs.remove(0));
                hidden = outputs.remove(0);
            }

            let output = output.unwrap();

            let target = Tensor::new_const(Matrix::new(batch_size, 1, batch[len - 1].clone()));

            let loss = criterion.forward(&output, &target);
            loss.backward(Tensor::new_const(Matrix::ones(1, 1)));

            optim.step(true);

            let current_loss = loss.0.borrow().data.data()[0];
            total_loss += current_loss;

            let current_accuracy: f64 = output
                .0
                .borrow()
                .data
                .row_iter()
                .zip(batch[len - 1].iter())
                .map(|(row, ix)| {
                    if argmax(row.raw_slice()) == (*ix) as usize {
                        1.0
                    } else {
                        0.0
                    }
                })
                .sum();

            total_accuracy += current_accuracy;
        }

        println!(
            "Loss: {}, Accuracy: {}",
            total_loss,
            total_accuracy / (total_data_size as f64)
        );
    }

    let batch = vec![
        vec![word_index[&"Mary".to_owned()] as f64],
        vec![word_index[&"moved".to_owned()] as f64],
        vec![word_index[&"to".to_owned()] as f64],
        vec![word_index[&"the".to_owned()] as f64],
    ];

    let mut hidden = model.create_start_state(1);
    let mut output = None;
    for row in batch.iter() {
        let input = Tensor::new_const(Matrix::new(1, 1, row.clone()));
        let rnn_input = embed.forward(&[&input]).remove(0);
        let mut outputs = model.forward(&[&rnn_input, &hidden]);
        output = Some(outputs.remove(0));
        hidden = outputs.remove(0);
    }

    let output = argmax(output.unwrap().0.borrow().data.row(0).raw_slice());
    println!("Prediction: {}", inverted_word_index[&output]);

    Ok(())
}
