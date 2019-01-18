use std::error::Error;
use std::ops::Mul;

use datasets::image::mnist;
use datasets::Dataset;
use indicatif::{ProgressBar, ProgressStyle};
use rand::distributions::Standard;
use rulinalg::matrix::{BaseMatrix, Matrix, MatrixSlice};

use grokking_deep_learning_rs::{
    argmax, generate_random_vector, sample_bernoulli_trials, softmax_mut, tanh_derivative, tanh_mut,
};

fn main() {
    println!("\nUpgrading our MNIST Network\n");
    mnist_tanh(0.5).unwrap();
}

#[allow(unused_doc_comments)]
fn mnist_tanh(keep_probability: f64) -> Result<(), Box<dyn Error>> {
    let (train_data, test_data) = mnist()?;

    let train_dataset_size = 1024;
    let test_dataset_size = 1024;

    let batch_size = 64; // 128 in the numpy version

    let (kernel_rows, kernel_cols) = (3, 3);
    let num_kernels = 4; // 16 in the numpy version

    let (train_images, train_labels) = process_mnist_filtered_dataset(
        train_data,
        train_dataset_size,
        batch_size,
        kernel_rows,
        kernel_cols,
    );

    let (test_images, test_labels) = process_mnist_filtered_dataset(
        test_data,
        test_dataset_size,
        batch_size,
        kernel_rows,
        kernel_cols,
    );

    let mut kernels = Matrix::new(
        kernel_rows * kernel_cols,
        num_kernels,
        generate_random_vector(
            kernel_rows * kernel_cols * num_kernels,
            0.02,
            -0.01,
            &Standard,
        ),
    );

    let mut weights_1_2 = Matrix::new(
        (28 - kernel_rows) * (28 - kernel_cols) * num_kernels,
        10,
        generate_random_vector(
            (28 - kernel_rows) * (28 - kernel_cols) * num_kernels * 10,
            0.2,
            -0.1,
            &Standard,
        ),
    );

    let alpha = 2.0;

    let iterations = 100;
    let progress = ProgressBar::new(iterations as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("{msg} {bar:40.cyan/blue} {pos:>7}/{len:7} [{elapsed_precise}]"),
    );

    for it in 0..iterations {
        let mut accuracy = 0.0;

        for (images, labels) in train_images.iter().zip(train_labels.iter()) {
            let labels =
                unsafe { MatrixSlice::from_raw_parts(labels.as_ptr(), batch_size, 10, 10) };

            let expanded_input_batch_size = images.len() * images[0].len();

            let expanded_input: Vec<f64> = images
                .iter()
                .flat_map(|kernel_inputs| kernel_inputs.into_iter())
                .flat_map(|kernel_inputs| kernel_inputs.into_iter().cloned())
                .collect();

            // [batch_size * 625, 9]
            let expanded_input = Matrix::new(
                expanded_input_batch_size,
                kernel_rows * kernel_cols,
                expanded_input,
            );

            // [batch_size * 625, 16]
            let kernel_output = (&expanded_input).mul(&kernels);

            // [batch_size, 625 * 16]
            // NOTE: this is the flatten step
            let mut hidden_layer = Matrix::new(
                batch_size,
                (28 - kernel_rows) * (28 - kernel_cols) * num_kernels,
                kernel_output.into_vec(),
            );

            /// Activation
            tanh_mut(&mut hidden_layer);

            /// Dropout
            let dropout_mask: Vec<f64> = sample_bernoulli_trials(
                keep_probability,
                batch_size * (28 - kernel_rows) * (28 - kernel_cols) * num_kernels,
            )
            .into_iter()
            .map(|v| v * (1.0 / keep_probability))
            .collect();

            let dropout_mask = Matrix::new(
                batch_size,
                (28 - kernel_rows) * (28 - kernel_cols) * num_kernels,
                dropout_mask,
            );

            let hidden_layer = hidden_layer.elemul(&dropout_mask);

            /// Final Outputs
            // [batch_size, 10]
            let mut predictions = (&hidden_layer).mul(&weights_1_2);
            softmax_mut(&mut predictions);

            /// NOTE: no error calculation still

            /// Accuracy
            for (r1, r2) in predictions.row_iter().zip(labels.row_iter()) {
                accuracy += if argmax(r1.raw_slice()) == argmax(r2.raw_slice()) {
                    1.0
                } else {
                    0.0
                }
            }

            /// delta_2_1
            let mut delta_2_1 = Matrix::new(batch_size, 10, vec![0.0; batch_size * 10]);
            for i in 0..batch_size {
                for j in 0..10 {
                    delta_2_1[[i, j]] =
                        (predictions[[i, j]] - labels[[i, j]]) / ((batch_size * batch_size) as f64);
                }
            }

            /// delta_1_0
            let mut delta_1_0 = (&delta_2_1)
                .mul(weights_1_2.transpose())
                .elemul(&tanh_derivative(&hidden_layer));

            for i in 0..batch_size {
                for j in 0..((28 - kernel_rows) * (28 - kernel_cols) * num_kernels) {
                    delta_1_0[[i, j]] *= dropout_mask[[i, j]];
                }
            }

            /// update weights_1_2
            let weight_delta_1_2 = hidden_layer.transpose().mul(delta_2_1);
            for i in 0..((28 - kernel_rows) * (28 - kernel_cols) * num_kernels) {
                for j in 0..10 {
                    weights_1_2[[i, j]] -= alpha * weight_delta_1_2[[i, j]];
                }
            }

            /// update weights_0_1
            // reorient delta_1_0
            let delta_1_0 = Matrix::new(
                batch_size * (28 - kernel_rows) * (28 - kernel_cols),
                num_kernels,
                delta_1_0.into_vec(),
            );

            let weight_delta_0_1 = expanded_input.transpose().mul(delta_1_0);
            for i in 0..(kernel_rows * kernel_cols) {
                for j in 0..num_kernels {
                    kernels[[i, j]] -= alpha * weight_delta_0_1[[i, j]];
                }
            }
        }

        let mut test_accuracy = 0.0;

        if (it + 1) % 10 == 0 {
            for (images, labels) in test_images.iter().zip(test_labels.iter()) {
                let labels =
                    unsafe { MatrixSlice::from_raw_parts(labels.as_ptr(), batch_size, 10, 10) };

                let expanded_input_batch_size = images.len() * images[0].len();

                let expanded_input: Vec<f64> = images
                    .iter()
                    .flat_map(|kernel_inputs| kernel_inputs.into_iter())
                    .flat_map(|kernel_inputs| kernel_inputs.into_iter().cloned())
                    .collect();

                // [batch_size * 625, 9]
                let expanded_input = Matrix::new(
                    expanded_input_batch_size,
                    kernel_rows * kernel_cols,
                    expanded_input,
                );

                // [batch_size * 625, 16]
                let kernel_output = expanded_input.mul(&kernels);

                // [batch_size, 625 * 16]
                // NOTE: this is the flatten step
                let mut hidden_layer = Matrix::new(
                    batch_size,
                    (28 - kernel_rows) * (28 - kernel_cols) * num_kernels,
                    kernel_output.into_vec(),
                );

                /// Activation
                tanh_mut(&mut hidden_layer);

                /// Dropout
                let dropout_mask: Vec<f64> = sample_bernoulli_trials(
                    keep_probability,
                    batch_size * (28 - kernel_rows) * (28 - kernel_cols) * num_kernels,
                )
                .into_iter()
                .map(|v| v * (1.0 / keep_probability))
                .collect();

                let dropout_mask = Matrix::new(
                    batch_size,
                    (28 - kernel_rows) * (28 - kernel_cols) * num_kernels,
                    dropout_mask,
                );

                let hidden_layer = hidden_layer.elemul(&dropout_mask);

                /// Final Outputs
                // [batch_size, 10]
                let mut predictions = hidden_layer.mul(&weights_1_2);
                softmax_mut(&mut predictions);

                /// NOTE: no error calculation still

                /// Accuracy
                for (r1, r2) in predictions.row_iter().zip(labels.row_iter()) {
                    test_accuracy += if argmax(r1.raw_slice()) == argmax(r2.raw_slice()) {
                        1.0
                    } else {
                        0.0
                    }
                }
            }

            progress.println(format!(
                "Iteration: {}, Train Accuracy: {}, Test Accuracy: {}",
                it + 1,
                accuracy / (train_dataset_size as f64),
                test_accuracy / (test_dataset_size as f64),
            ));
        }

        progress.inc(1);
        progress.set_message(&format!(
            "Train Accuracy: {}",
            accuracy / (train_dataset_size as f64),
        ));
    }

    Ok(())
}

fn process_mnist_filtered_dataset(
    dataset: impl Dataset<Item = (Vec<u8>, u8)>,
    dataset_size: usize,
    batch_size: usize,
    kernel_rows: usize,
    kernel_cols: usize,
) -> (Vec<Vec<Vec<Vec<f64>>>>, Vec<Vec<f64>>) {
    let (images, labels): (Vec<Vec<u8>>, Vec<u8>) = dataset.take(dataset_size).unzip();

    // extract kernel sized image sections from images
    // [_, batch, kernels, kernel_image]
    let images = images
        .into_iter()
        .map(|img| {
            // convert each image into a vectors of kernel inputs of size 3x3

            let mut kernel_inputs = Vec::with_capacity((28 - kernel_rows) * (28 - kernel_cols));

            for i in 0..(28 - kernel_rows) {
                for j in 0..(28 - kernel_cols) {
                    let mut kernel_input = vec![0.0; kernel_rows * kernel_cols];

                    for k in 0..kernel_rows {
                        for l in 0..kernel_cols {
                            kernel_input[k * kernel_cols + l] = img[(i + k) * 28 + (j + l)] as f64;
                        }
                    }

                    kernel_inputs.push(kernel_input);
                }
            }

            kernel_inputs
        })
        .batch(batch_size, false)
        .collect();

    // [_, batch, label]
    let labels = labels
        .into_iter()
        .map(|l| {
            let mut v = vec![0.0; 10];
            v[l as usize] = 1.0;
            v
        })
        .batch(batch_size, false)
        // flatten each batch so it can be converted to MatrixSlice easily
        .map(|b| b.into_iter().flat_map(|v| v.into_iter()).collect())
        .collect();

    (images, labels)
}
