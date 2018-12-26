//! Grokking Deep Learning - Chapter 3
//!
//! https://github.com/iamtrask/Grokking-Deep-Learning/blob/master/Chapter3%20-%20%20Forward%20Propagation%20-%20Intro%20to%20Neural%20Prediction.ipynb

extern crate grokking_deep_learning_rs;

use grokking_deep_learning_rs::{
    dot, elementwise_scalar_multiplication, matrix_vector_multiplication, Matrix, Vector,
};

fn main() {
    // different sections of the chapter in order.
    what_is_a_neural_network();
    making_a_prediction_with_multiple_inputs();
    making_a_prediction_with_multiple_outputs();
    predicting_with_multiple_inputs_and_outputs();
    predicting_on_predictions();
}

/// A Simple Neural Network making a prediction
///
/// What is a neural network?

fn what_is_a_neural_network() {
    let number_of_toes = vec![8.5, 9.5, 10.0, 9.0];

    let input = number_of_toes[0];
    let weight = 0.1;

    let prediction = neural_network_1(input, weight);
    println!("prediction: {}", prediction);
}

fn neural_network_1(input: f64, weight: f64) -> f64 {
    let prediction = input * weight;
    prediction
}

/// Making a prediction with multiple inputs

fn making_a_prediction_with_multiple_inputs() {
    let toes = vec![8.5, 9.5, 9.9, 9.0];
    let wlrec = vec![0.65, 0.8, 0.8, 0.9];
    let nfans = vec![1.2, 1.3, 0.5, 1.0];

    let input = vec![toes[0], wlrec[0], nfans[0]];
    let weights = vec![0.1, 0.2, 0.0];

    let pred = neural_network_2(input, weights);
    println!("prediction: {}", pred);
}

fn neural_network_2(input: Vec<f64>, weights: Vec<f64>) -> f64 {
    dot(input, &weights)
}

/// Making a prediction with multiple outputs

fn making_a_prediction_with_multiple_outputs() {
    let wlrec = vec![0.65, 0.8, 0.8, 0.9];

    let input = wlrec[0];
    let weights = vec![0.3, 0.2, 0.9];

    let pred = neural_network_3(input, weights);
    println!("predictions: {:?}", pred);
}

fn neural_network_3(input: f64, weights: Vec<f64>) -> Vec<f64> {
    elementwise_scalar_multiplication(weights, input)
}

/// Predicting with multiple inputs and outputs

fn predicting_with_multiple_inputs_and_outputs() {
    let toes = vec![8.5, 9.5, 9.9, 9.0];
    let wlrec = vec![0.65, 0.8, 0.8, 0.9];
    let nfans = vec![1.2, 1.3, 0.5, 1.0];

    let input = vec![toes[0], wlrec[0], nfans[0]];
    let weights = vec![
        vec![0.1, 0.1, -0.3],
        vec![0.1, 0.2, 0.0],
        vec![0.0, 1.3, 0.1],
    ];

    let pred = neural_network_4(input, weights);
    println!("predictions: {:?}", pred);
}

fn neural_network_4(input: Vector, weights: Matrix) -> Vector {
    matrix_vector_multiplication(weights, &input)
}

/// Predicting on Predictions

fn predicting_on_predictions() {
    let toes = vec![8.5, 9.5, 9.9, 9.0];
    let wlrec = vec![0.65, 0.8, 0.8, 0.9];
    let nfans = vec![1.2, 1.3, 0.5, 1.0];

    let input = vec![toes[0], wlrec[0], nfans[0]];
    let input_weights = vec![
        vec![0.1, 0.2, -0.1],
        vec![-0.1, 0.1, 0.9],
        vec![0.1, 0.4, 0.1],
    ];
    let hidden1_weights = vec![
        vec![0.3, 1.1, -0.3],
        vec![0.1, 0.2, 0.0],
        vec![0.0, 1.3, 0.1],
    ];

    let pred = neural_network_5(input, input_weights, hidden1_weights);
    println!("predictions: {:?}", pred);
}

fn neural_network_5(input: Vector, input_weights: Matrix, hidden1_weights: Matrix) -> Vector {
    matrix_vector_multiplication(
        hidden1_weights,
        &matrix_vector_multiplication(input_weights, &input),
    )
}
