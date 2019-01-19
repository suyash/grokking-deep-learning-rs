//! Chapter9 - Intro to Activation Functions - Modeling Probabilities.ipynb
//!
//! https://github.com/iamtrask/Grokking-Deep-Learning/blob/master/Chapter9%20-%20Intro%20to%20Activation%20Functions%20-%20Modeling%20Probabilities.ipynb

use std::error::Error;
use std::ops::Mul;

use datasets::image::mnist;
use indicatif::{ProgressBar, ProgressStyle};
use rand::distributions::Standard;
use rulinalg::matrix::{BaseMatrix, Matrix, MatrixSlice};

use grokking_deep_learning_rs::{
    argmax, generate_random_vector, process_mnist_batch_dataset, sample_bernoulli_trials,
    softmax_mut, tanh_derivative, tanh_mut,
};

fn main() {
    println!("\nUpgrading our MNIST Network\n");
    mnist_tanh(0.5).unwrap();
}

fn mnist_tanh(keep_probability: f64) -> Result<(), Box<dyn Error>> {
    let (train_data, test_data) = mnist()?;

    let train_data_size = 1000;
    let test_data_size = 1000;
    let batch_size = 100;

    let (train_images, train_labels) =
        process_mnist_batch_dataset(train_data, train_data_size, batch_size);
    let (test_images, test_labels) =
        process_mnist_batch_dataset(test_data, test_data_size, batch_size);

    let (alpha, hidden_size) = (2.0, 100);

    let mut weights_0_1 = Matrix::new(
        784,
        hidden_size,
        generate_random_vector(784 * hidden_size, 0.02, -0.01, &Standard),
    );
    let mut weights_1_2 = Matrix::new(
        hidden_size,
        10,
        generate_random_vector(hidden_size * 10, 0.2, -0.1, &Standard),
    );

    let iterations = 100;
    let progress = ProgressBar::new(iterations as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("{msg} {bar:40.cyan/blue} {pos:>7}/{len:7} [{elapsed_precise}]"),
    );

    for it in 0..iterations {
        let mut accuracy = 0.0;

        for (images, labels) in train_images.iter().zip(train_labels.iter()) {
            let images =
                unsafe { MatrixSlice::from_raw_parts(images.as_ptr(), batch_size, 784, 784) };
            let labels =
                unsafe { MatrixSlice::from_raw_parts(labels.as_ptr(), batch_size, 10, 10) };

            let mut hidden_layer = (&images).mul(&weights_0_1);
            tanh_mut(&mut hidden_layer);

            let dropout_mask = Matrix::new(
                batch_size,
                hidden_size,
                sample_bernoulli_trials(keep_probability, batch_size * hidden_size),
            );

            for i in 0..batch_size {
                for j in 0..hidden_size {
                    hidden_layer[[i, j]] *= dropout_mask[[i, j]] * (1.0 / keep_probability);
                }
            }

            let mut outputs = (&hidden_layer).mul(&weights_1_2);
            softmax_mut(&mut outputs);

            for (r, l) in (&outputs).row_iter().zip(labels.row_iter()) {
                accuracy += if argmax(r.raw_slice()) == argmax(l.raw_slice()) {
                    1.0
                } else {
                    0.0
                }
            }

            // NOTE: no error calc here
            // just taking on faith that the derivative for the final layer = (value - true_value) / (batch_size^2)

            let mut delta_2_1 = Matrix::zeros(batch_size, 10);
            for i in 0..batch_size {
                for j in 0..10 {
                    delta_2_1[[i, j]] =
                        (outputs[[i, j]] - labels[[i, j]]) / ((batch_size * batch_size) as f64);
                }
            }

            let mut delta_1_0 = (&delta_2_1)
                .mul(weights_1_2.transpose())
                .elemul(&tanh_derivative(&hidden_layer));

            for i in 0..batch_size {
                for j in 0..hidden_size {
                    delta_1_0[[i, j]] *= dropout_mask[[i, j]];
                }
            }

            let weight_delta_1_2 = hidden_layer.transpose().mul(delta_2_1);
            let weight_delta_0_1 = images.transpose().mul(delta_1_0);

            for i in 0..hidden_size {
                for k in 0..10 {
                    weights_1_2[[i, k]] -= alpha * weight_delta_1_2[[i, k]];
                }
            }

            for i in 0..784 {
                for k in 0..hidden_size {
                    weights_0_1[[i, k]] -= alpha * weight_delta_0_1[[i, k]];
                }
            }
        }

        if (it + 1) % 10 == 0 {
            let mut test_accuracy = 0.0;

            for (images, labels) in test_images.iter().zip(test_labels.iter()) {
                let images =
                    unsafe { MatrixSlice::from_raw_parts(images.as_ptr(), batch_size, 784, 784) };
                let labels =
                    unsafe { MatrixSlice::from_raw_parts(labels.as_ptr(), batch_size, 10, 10) };

                let mut hidden_layer = images.mul(&weights_0_1);
                tanh_mut(&mut hidden_layer);

                let mut outputs = hidden_layer.mul(&weights_1_2);
                softmax_mut(&mut outputs);

                for (r, l) in (&outputs).row_iter().zip(labels.row_iter()) {
                    test_accuracy += if argmax(r.raw_slice()) == argmax(l.raw_slice()) {
                        1.0
                    } else {
                        0.0
                    }
                }
            }

            progress.println(format!(
                "Iteration: {}, Train Accuracy: {}, Test Accuracy: {}",
                it + 1,
                accuracy / (train_data_size as f64),
                test_accuracy / (test_data_size as f64),
            ));
        }

        progress.inc(1);
        progress.set_message(&format!(
            "Train Accuracy: {}",
            accuracy / (train_data_size as f64),
        ));
    }

    progress.finish_and_clear();

    Ok(())
}
