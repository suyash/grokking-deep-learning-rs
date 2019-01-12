use std::error::Error;
use std::ops::Mul;

use datasets::image::mnist;
use datasets::Dataset;
use grokking_deep_learning_rs::*;
use rand::distributions::{Bernoulli, Distribution, Standard};
use rand::{thread_rng, Rng};
use rulinalg::matrix::{BaseMatrix, Matrix, MatrixSlice};

fn main() {
    println!("\n3 Layer Network on MNIST\n");
    three_layer_mnist().unwrap();

    println!("\n3 Layer Network on MNIST with validation every 10 iterations\n");
    three_layer_mnist_with_validation().unwrap();

    println!("\nDropout\n");
    three_layer_mnist_with_validation_and_dropout(0.3).unwrap();

    println!("\nBatched Gradient Descent with Dropout\n");
    batched_gradient_descent_with_dropout(0.5).unwrap();
}

fn three_layer_mnist() -> Result<(), Box<dyn Error>> {
    let dataset_size = 100; // 1000 in notebook with numpy
    let test_dataset_size = 10000;

    let (train_data, test_data) = mnist()?;

    let (images, labels): (Vec<_>, Vec<_>) = train_data.take(dataset_size).unzip();

    let images: Vec<Vec<f64>> = images
        .iter()
        .map(|img| img.iter().map(|v| (*v as f64) / 255.0).collect())
        .collect();

    let labels: Vec<Vec<f64>> = labels
        .iter()
        .map(|l| {
            let mut v = vec![0.0; 10];
            v[*l as usize] = 1.0;
            v
        })
        .collect();

    let (alpha, hidden_size) = (0.005, 40);

    let iterations = 100; // NOTE: cannot run this for 350 iterations because of slower matrix multiplication.

    let mut weights_0_1 = Matrix::new(784, hidden_size, random_matrix(784, hidden_size, &Standard));
    let mut weights_1_2 = Matrix::new(hidden_size, 10, random_matrix(hidden_size, 10, &Standard));

    // Training

    for it in 0..iterations {
        let mut total_error = 0.0;
        let mut accuracy = 0.0;

        for (image, label) in images.iter().zip(labels.iter()) {
            let image = unsafe { MatrixSlice::from_raw_parts(image.as_ptr(), 1, 784, 1) };

            let mut hidden_layer = (&image).mul(&weights_0_1);
            for j in 0..hidden_size {
                if hidden_layer[[0, j]] < 0.0 {
                    hidden_layer[[0, j]] = 0.0;
                }
            }

            let output = (&hidden_layer).mul(&weights_1_2);

            accuracy += if argmax(&label) == argmax(output.data()) {
                1.0
            } else {
                0.0
            };

            let error: f64 = output
                .data()
                .iter()
                .zip(label.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum();

            total_error += error;

            let delta_2_1 = output - Matrix::new(1, 10, label.clone());

            let mut relu_deriv = Matrix::new(1, hidden_size, vec![0.0; hidden_size]);
            for i in 0..hidden_size {
                if hidden_layer[[0, i]] >= 0.0 {
                    relu_deriv[[0, i]] = 1.0;
                }
            }

            let delta_1_0 = (&delta_2_1)
                .mul(weights_1_2.transpose())
                .elemul(&relu_deriv);

            let weight_delta_1_2 = hidden_layer.transpose().mul(delta_2_1);

            // avoid another clone of image
            let weight_delta_0_1 = image.transpose().mul(delta_1_0);

            for (i, x) in weights_0_1.mut_data().into_iter().enumerate() {
                *x -= alpha * weight_delta_0_1.data()[i];
            }

            for (i, x) in weights_1_2.mut_data().into_iter().enumerate() {
                *x -= alpha * weight_delta_1_2.data()[i];
            }
        }

        println!(
            "Iteration: {}, Train Accuracy: {}, Train Error: {}",
            it + 1,
            accuracy / (dataset_size as f64),
            total_error / (dataset_size as f64)
        );
    }

    // Inference

    println!("Evaluating on the test dataset");

    let (images, labels): (Vec<_>, Vec<_>) = test_data.take(test_dataset_size).unzip();

    let images: Vec<Vec<f64>> = images
        .into_iter()
        .map(|img| img.into_iter().map(|v| (v as f64) / 255.0).collect())
        .collect();

    let labels: Vec<Vec<f64>> = labels
        .into_iter()
        .map(|l| {
            let mut v = vec![0.0; 10];
            v[l as usize] = 1.0;
            v
        })
        .collect();

    let mut total_error = 0.0;
    let mut accuracy = 0.0;

    for (image, label) in images.into_iter().zip(labels.into_iter()) {
        let image = Matrix::new(1, 784, image);

        let mut hidden_layer = image.mul(&weights_0_1);

        // relu
        for j in 0..hidden_size {
            if hidden_layer[[0, j]] < 0.0 {
                hidden_layer[[0, j]] = 0.0;
            }
        }

        let output = hidden_layer.mul(&weights_1_2);

        accuracy += if argmax(&label) == argmax(output.data()) {
            1.0
        } else {
            0.0
        };

        let error: f64 = output
            .iter()
            .zip(label.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum();

        total_error += error;
    }

    println!(
        "\nTest Accuracy: {}, Test Error: {}\n",
        accuracy / (test_dataset_size as f64),
        total_error / (test_dataset_size as f64),
    );

    Ok(())
}

fn three_layer_mnist_with_validation() -> Result<(), Box<dyn Error>> {
    let dataset_size = 100; // 1000 in notebook with numpy
    let test_dataset_size = 1000;

    let (train_data, test_data) = mnist()?;

    let (images, labels): (Vec<_>, Vec<_>) = train_data.take(dataset_size).unzip();

    let images: Vec<Vec<f64>> = images
        .iter()
        .map(|img| img.iter().map(|v| (*v as f64) / 255.0).collect())
        .collect();

    let labels: Vec<Vec<f64>> = labels
        .iter()
        .map(|l| {
            let mut v = vec![0.0; 10];
            v[*l as usize] = 1.0;
            v
        })
        .collect();

    let (alpha, hidden_size) = (0.005, 40);

    let (test_images, test_labels): (Vec<_>, Vec<_>) = test_data.take(test_dataset_size).unzip();

    let test_images: Vec<Vec<f64>> = test_images
        .into_iter()
        .map(|img| img.into_iter().map(|v| (v as f64) / 255.0).collect())
        .collect();

    let test_labels: Vec<Vec<f64>> = test_labels
        .into_iter()
        .map(|l| {
            let mut v = vec![0.0; 10];
            v[l as usize] = 1.0;
            v
        })
        .collect();

    let iterations = 100; // NOTE: cannot run this for 350 iterations because of slower matrix multiplication.

    let mut weights_0_1 = Matrix::new(784, hidden_size, random_matrix(784, hidden_size, &Standard));
    let mut weights_1_2 = Matrix::new(hidden_size, 10, random_matrix(hidden_size, 10, &Standard));

    // Training

    for it in 0..iterations {
        let mut total_error = 0.0;
        let mut accuracy = 0.0;

        for (image, label) in images.iter().zip(labels.iter()) {
            let image = unsafe { MatrixSlice::from_raw_parts(image.as_ptr(), 1, 784, 1) };

            let mut hidden_layer = (&image).mul(&weights_0_1);
            for j in 0..hidden_size {
                if hidden_layer[[0, j]] < 0.0 {
                    hidden_layer[[0, j]] = 0.0;
                }
            }

            let output = (&hidden_layer).mul(&weights_1_2);

            accuracy += if argmax(&label) == argmax(output.data()) {
                1.0
            } else {
                0.0
            };

            let error: f64 = output
                .data()
                .iter()
                .zip(label.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum();

            total_error += error;

            let delta_2_1 = output - Matrix::new(1, 10, label.clone());

            let mut relu_deriv = Matrix::new(1, hidden_size, vec![0.0; hidden_size]);
            for i in 0..hidden_size {
                if hidden_layer[[0, i]] >= 0.0 {
                    relu_deriv[[0, i]] = 1.0;
                }
            }

            let delta_1_0 = (&delta_2_1)
                .mul(weights_1_2.transpose())
                .elemul(&relu_deriv);

            let weight_delta_1_2 = hidden_layer.transpose().mul(delta_2_1);

            // avoid another clone of image
            let weight_delta_0_1 = image.transpose().mul(delta_1_0);

            for (i, x) in weights_0_1.mut_data().into_iter().enumerate() {
                *x -= alpha * weight_delta_0_1.data()[i];
            }

            for (i, x) in weights_1_2.mut_data().into_iter().enumerate() {
                *x -= alpha * weight_delta_1_2.data()[i];
            }
        }

        if (it + 1) % 10 == 0 {
            // Inference

            let mut total_test_error = 0.0;
            let mut test_accuracy = 0.0;

            for (image, label) in test_images.iter().zip(test_labels.iter()) {
                let image = unsafe { MatrixSlice::from_raw_parts(image.as_ptr(), 1, 784, 1) };

                let mut hidden_layer = image.mul(&weights_0_1);

                // relu
                for j in 0..hidden_size {
                    if hidden_layer[[0, j]] < 0.0 {
                        hidden_layer[[0, j]] = 0.0;
                    }
                }

                let output = hidden_layer.mul(&weights_1_2);

                test_accuracy += if argmax(&label) == argmax(output.data()) {
                    1.0
                } else {
                    0.0
                };

                let error: f64 = output
                    .iter()
                    .zip(label.iter())
                    .map(|(p, t)| (p - t).powi(2))
                    .sum();

                total_test_error += error;
            }

            println!(
                "\nIteration: {}, Train Accuracy: {}, Train Error: {}, Test Accuracy: {}, Test Error: {}\n",
                it + 1,
                accuracy / (dataset_size as f64),
                total_error / (dataset_size as f64),
                test_accuracy / (test_dataset_size as f64),
                total_test_error / (test_dataset_size as f64),
            );
        } else {
            println!(
                "Iteration: {}, Train Accuracy: {}, Train Error: {}",
                it + 1,
                accuracy / (dataset_size as f64),
                total_error / (dataset_size as f64)
            );
        }
    }

    Ok(())
}

fn three_layer_mnist_with_validation_and_dropout(
    keep_probability: f64,
) -> Result<(), Box<dyn Error>> {
    let dataset_size = 1000; // 1000 in notebook with numpy
    let test_dataset_size = 1000;

    let (train_data, test_data) = mnist()?;

    let (images, labels): (Vec<_>, Vec<_>) = train_data.take(dataset_size).unzip();

    let images: Vec<Vec<f64>> = images
        .iter()
        .map(|img| img.iter().map(|v| (*v as f64) / 255.0).collect())
        .collect();

    let labels: Vec<Vec<f64>> = labels
        .iter()
        .map(|l| {
            let mut v = vec![0.0; 10];
            v[*l as usize] = 1.0;
            v
        })
        .collect();

    let (alpha, hidden_size) = (0.005, 40);

    let (test_images, test_labels): (Vec<_>, Vec<_>) = test_data.take(test_dataset_size).unzip();

    let test_images: Vec<Vec<f64>> = test_images
        .into_iter()
        .map(|img| img.into_iter().map(|v| (v as f64) / 255.0).collect())
        .collect();

    let test_labels: Vec<Vec<f64>> = test_labels
        .into_iter()
        .map(|l| {
            let mut v = vec![0.0; 10];
            v[l as usize] = 1.0;
            v
        })
        .collect();

    let iterations = 100; // NOTE: cannot run this for 350 iterations because of slower matrix multiplication.

    let mut weights_0_1 = Matrix::new(784, hidden_size, random_matrix(784, hidden_size, &Standard));
    let mut weights_1_2 = Matrix::new(hidden_size, 10, random_matrix(hidden_size, 10, &Standard));

    let dist = Bernoulli::new(keep_probability);

    // Training

    for it in 0..iterations {
        let mut total_error = 0.0;
        let mut accuracy = 0.0;

        for (image, label) in images.iter().zip(labels.iter()) {
            let image = unsafe { MatrixSlice::from_raw_parts(image.as_ptr(), 1, 784, 1) };

            let mut hidden_layer = (&image).mul(&weights_0_1);
            for j in 0..hidden_size {
                if hidden_layer[[0, j]] < 0.0 {
                    hidden_layer[[0, j]] = 0.0;
                }
            }

            let dropout_mask_data: Vec<f64> = thread_rng()
                .sample_iter(&dist)
                .take(hidden_size)
                .map(|v| if v { 1.0 } else { 0.0 })
                .collect();

            let dropout_mask = Matrix::new(1, hidden_size, dropout_mask_data);

            for j in 0..hidden_size {
                hidden_layer[[0, j]] *= dropout_mask[[0, j]] * (1.0 / keep_probability);
            }

            let output = (&hidden_layer).mul(&weights_1_2);

            accuracy += if argmax(&label) == argmax(output.data()) {
                1.0
            } else {
                0.0
            };

            let error: f64 = output
                .data()
                .iter()
                .zip(label.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum();

            total_error += error;

            let delta_2_1 = output - Matrix::new(1, 10, label.clone());

            let mut relu_deriv = Matrix::new(1, hidden_size, vec![0.0; hidden_size]);
            for i in 0..hidden_size {
                if hidden_layer[[0, i]] >= 0.0 {
                    relu_deriv[[0, i]] = 1.0;
                }
            }

            let mut delta_1_0 = (&delta_2_1)
                .mul(weights_1_2.transpose())
                .elemul(&relu_deriv);

            for j in 0..hidden_size {
                delta_1_0[[0, j]] *= dropout_mask[[0, j]] * (1.0 / keep_probability);
            }

            let weight_delta_1_2 = hidden_layer.transpose().mul(delta_2_1);

            // avoid another clone of image
            let weight_delta_0_1 = image.transpose().mul(delta_1_0);

            for (i, x) in weights_0_1.mut_data().into_iter().enumerate() {
                *x -= alpha * weight_delta_0_1.data()[i];
            }

            for (i, x) in weights_1_2.mut_data().into_iter().enumerate() {
                *x -= alpha * weight_delta_1_2.data()[i];
            }
        }

        if (it + 1) % 10 == 0 {
            // Inference

            let mut total_test_error = 0.0;
            let mut test_accuracy = 0.0;

            for (image, label) in test_images.iter().zip(test_labels.iter()) {
                let image = unsafe { MatrixSlice::from_raw_parts(image.as_ptr(), 1, 784, 1) };

                let mut hidden_layer = image.mul(&weights_0_1);

                // relu
                for j in 0..hidden_size {
                    if hidden_layer[[0, j]] < 0.0 {
                        hidden_layer[[0, j]] = 0.0;
                    }
                }

                let output = hidden_layer.mul(&weights_1_2);

                test_accuracy += if argmax(&label) == argmax(output.data()) {
                    1.0
                } else {
                    0.0
                };

                let error: f64 = output
                    .iter()
                    .zip(label.iter())
                    .map(|(p, t)| (p - t).powi(2))
                    .sum();

                total_test_error += error;
            }

            println!(
                "\nIteration: {}, Train Accuracy: {}, Train Error: {}, Test Accuracy: {}, Test Error: {}\n",
                it + 1,
                accuracy / (dataset_size as f64),
                total_error / (dataset_size as f64),
                test_accuracy / (test_dataset_size as f64),
                total_test_error / (test_dataset_size as f64),
            );
        } else {
            println!(
                "Iteration: {}, Train Accuracy: {}, Train Error: {}",
                it + 1,
                accuracy / (dataset_size as f64),
                total_error / (dataset_size as f64)
            );
        }
    }

    Ok(())
}

fn batched_gradient_descent_with_dropout(keep_probability: f64) -> Result<(), Box<dyn Error>> {
    let dataset_size = 1000; // 1000 in notebook with numpy
    let test_dataset_size = 1000;

    let batch_size = 100;

    let (train_data, test_data) = mnist()?;

    let (images, labels) = process_batch_dataset(train_data, dataset_size, batch_size);
    let (test_images, test_labels) =
        process_batch_dataset(test_data, test_dataset_size, batch_size);

    let (alpha, hidden_size) = (0.001, 40);

    let iterations = 100; // NOTE: cannot run this for 350 iterations because of slower matrix multiplication.

    let mut weights_0_1 = Matrix::new(784, hidden_size, random_matrix(784, hidden_size, &Standard));
    let mut weights_1_2 = Matrix::new(hidden_size, 10, random_matrix(hidden_size, 10, &Standard));

    let dist = Bernoulli::new(keep_probability);

    // Training

    for it in 0..iterations {
        let mut total_error = 0.0;
        let mut accuracy = 0.0;

        for (image, label) in images.iter().zip(labels.iter()) {
            let image = unsafe { MatrixSlice::from_raw_parts(image.as_ptr(), batch_size, 784, 784) };
            let label = unsafe { MatrixSlice::from_raw_parts(label.as_ptr(), batch_size, 10, 10) };

            let mut hidden_layer = (&image).mul(&weights_0_1);
            for i in 0..batch_size {
                for j in 0..hidden_size {
                    if hidden_layer[[i, j]] < 0.0 {
                        hidden_layer[[i, j]] = 0.0;
                    }
                }
            }

            let dropout_mask_data: Vec<f64> = thread_rng()
                .sample_iter(&dist)
                .take(batch_size * hidden_size)
                .map(|v| if v { 1.0 } else { 0.0 })
                .collect();

            let dropout_mask = Matrix::new(batch_size, hidden_size, dropout_mask_data);

            for i in 0..batch_size {
                for j in 0..hidden_size {
                    hidden_layer[[i, j]] *= dropout_mask[[i, j]] * (1.0 / keep_probability);
                }
            }

            let outputs = (&hidden_layer).mul(&weights_1_2);

            for (output, l) in outputs.row_iter().zip(label.row_iter()) {
                if argmax(output.raw_slice()) == argmax(l.raw_slice()) {
                    accuracy += 1.0;
                }
            }

            for (output, l) in outputs.row_iter().zip(label.row_iter()) {
                let err: f64 = output
                    .raw_slice()
                    .iter()
                    .zip(l.raw_slice().iter())
                    .map(|(p, t)| (p - t).powi(2))
                    .sum();
                total_error += err;
            }

            let mut delta_2_1 = Matrix::new(batch_size, 10, vec![0.0; batch_size * 10]);
            for i in 0..batch_size {
                for j in 0..10 {
                    delta_2_1[[i, j]] = outputs[[i, j]] - label[[i, j]];
                }
            }

            let mut relu_deriv = Matrix::new(batch_size, hidden_size, vec![0.0; batch_size * hidden_size]);
            for i in 0..batch_size {
                for j in 0..hidden_size {
                    if hidden_layer[[i, j]] >= 0.0 {
                        relu_deriv[[i, j]] = 1.0;
                    }
                }
            }

            let mut delta_1_0 = (&delta_2_1)
                .mul(weights_1_2.transpose())
                .elemul(&relu_deriv);

            for i in 0..batch_size {
                for j in 0..hidden_size {
                    delta_1_0[[i, j]] *= dropout_mask[[i, j]] * (1.0 / keep_probability);
                }
            }

            let weight_delta_1_2 = hidden_layer.transpose().mul(delta_2_1);
            let weight_delta_0_1 = image.transpose().mul(delta_1_0);

            for (i, x) in weights_0_1.mut_data().into_iter().enumerate() {
                *x -= alpha * weight_delta_0_1.data()[i];
            }

            for (i, x) in weights_1_2.mut_data().into_iter().enumerate() {
                *x -= alpha * weight_delta_1_2.data()[i];
            }
        }

        if (it + 1) % 10 == 0 {
            // Inference

            let mut total_test_error = 0.0;
            let mut test_accuracy = 0.0;

            for (image, label) in test_images.iter().zip(test_labels.iter()) {
                let image =
                    unsafe { MatrixSlice::from_raw_parts(image.as_ptr(), batch_size, 784, 784) };
                let label =
                    unsafe { MatrixSlice::from_raw_parts(label.as_ptr(), batch_size, 10, 10) };

                let mut hidden_layer = image.mul(&weights_0_1);
                for i in 0..batch_size {
                    for j in 0..hidden_size {
                        if hidden_layer[[i, j]] < 0.0 {
                            hidden_layer[[i, j]] = 0.0;
                        }
                    }
                }

                let outputs = hidden_layer.mul(&weights_1_2);

                for (output, l) in outputs.row_iter().zip(label.row_iter()) {
                    if argmax(output.raw_slice()) == argmax(l.raw_slice()) {
                        test_accuracy += 1.0;
                    }
                }

                for (output, l) in outputs.row_iter().zip(label.row_iter()) {
                    let err: f64 = output
                        .raw_slice()
                        .iter()
                        .zip(l.raw_slice().iter())
                        .map(|(p, t)| (p - t).powi(2))
                        .sum();

                    total_test_error += err;
                }
            }

            println!(
                "\nIteration: {}, Train Accuracy: {}, Train Error: {}, Test Accuracy: {}, Test Error: {}\n",
                it + 1,
                accuracy / (dataset_size as f64),
                total_error / (dataset_size as f64),
                test_accuracy / (test_dataset_size as f64),
                total_test_error / (test_dataset_size as f64),
            );
        } else {
            println!(
                "Iteration: {}, Train Accuracy: {}, Train Error: {}",
                it + 1,
                accuracy / (dataset_size as f64),
                total_error / (dataset_size as f64)
            );
        }
    }

    Ok(())
}

fn process_batch_dataset(
    dataset: impl Dataset<Item = (Vec<u8>, u8)>,
    dataset_size: usize,
    batch_size: usize,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let normalize_image = |img: Vec<u8>| img.iter().map(|v| (*v as f64) / 255.0).collect();
    let encode_label = |l| {
        let mut v = vec![0.0; 10];
        v[l as usize] = 1.0;
        v
    };

    let (images, labels): (Vec<_>, Vec<_>) = dataset
        .take(dataset_size)
        .map(|(i, l)| (normalize_image(i), encode_label(l)))
        .unzip();

    let images = images
        .into_iter()
        .batch(batch_size)
        .map(|v| {
            v.into_iter()
                .fold(Vec::with_capacity(batch_size * 784), |mut acc, mut img| {
                    acc.append(&mut img);
                    acc
                })
        })
        .collect();

    let labels = labels
        .into_iter()
        .batch(batch_size)
        .map(|v| {
            v.into_iter()
                .fold(Vec::with_capacity(batch_size * 10), |mut acc, mut l| {
                    acc.append(&mut l);
                    acc
                })
        })
        .collect();

    (images, labels)
}

fn random_matrix(rows: usize, columns: usize, dist: &impl Distribution<f64>) -> Vector {
    (0..(rows * columns))
        .map(|_| 0.2 * thread_rng().sample(dist) - 0.1)
        .collect()
}
