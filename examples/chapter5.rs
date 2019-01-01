use grokking_deep_learning_rs::{
    dot, elementwise_scalar_multiplication, matrix_vector_multiplication, Matrix, Vector,
};

fn main() {
    println!("\nGradient Descent Learning with Multiple Inputs.\n");
    gradient_descent_with_multiple_inputs();

    println!("\nLet's Watch Several Steps of Learning\n");
    gradient_descent_with_multiple_inputs_iterations();

    println!("\nFreezing one weight, What does it do?\n");
    gradient_descent_with_multiple_inputs_frozen_weights();

    println!("\nGradient Descent Learning with multiple outputs\n");
    gradient_descent_with_multiple_outputs();

    println!("\nGradient Descent with multiple inputs and outputs\n");
    gradient_descent_with_multiple_inputs_and_outputs();
}

/// Gradient Descent Learning with Multiple Inputs.

fn gradient_descent_with_multiple_inputs() {
    let mut weights: Vector = vec![0.1, 0.2, -0.1];

    let toes = vec![8.5, 9.5, 9.9, 9.0];
    let wlrec = vec![0.65, 0.8, 0.8, 0.9];
    let nfans = vec![1.2, 1.3, 0.5, 1.0];

    let input = vec![toes[0], wlrec[0], nfans[0]];

    let win_or_lose_binary = [1.0, 1.0, 0.0, 1.0];
    let truth = win_or_lose_binary[0];

    let pred = neural_network_1(input.clone(), &weights);
    let error = (pred - truth).powf(2.0);
    println!("Error: {}, Prediction: {}", error, pred);

    let delta = pred - truth;
    let weight_delta = elementwise_scalar_multiplication(input, delta);

    let alpha = 0.01;
    for i in 0..3 {
        weights[i] -= alpha * weight_delta[i];
    }
    println!("Weights: {:?}, Weight Deltas: {:?}", weights, weight_delta);
}

fn neural_network_1(input: Vector, weights: &[f64]) -> f64 {
    dot(input, weights)
}

/// Let's Watch Several Steps of Learning

fn gradient_descent_with_multiple_inputs_iterations() {
    let mut weights: Vector = vec![0.1, 0.2, -0.1];

    let toes = vec![8.5, 9.5, 9.9, 9.0];
    let wlrec = vec![0.65, 0.8, 0.8, 0.9];
    let nfans = vec![1.2, 1.3, 0.5, 1.0];

    let input = vec![toes[0], wlrec[0], nfans[0]];

    let win_or_lose_binary = [1.0, 1.0, 0.0, 1.0];
    let truth = win_or_lose_binary[0];

    let alpha = 0.01;

    for i in 0..3 {
        println!("Iteration {}", i + 1);

        let pred = neural_network_1(input.clone(), &weights);
        let error = (pred - truth).powf(2.0);
        println!("Error: {}, Prediction: {}", error, pred);

        let delta = pred - truth;
        let weight_delta = elementwise_scalar_multiplication(input.clone(), delta);

        for i in 0..3 {
            weights[i] -= alpha * weight_delta[i];
        }
        println!(
            "Weights: {:?}, Weight Deltas: {:?}\n",
            weights, weight_delta
        );
    }
}

/// Freezing one weight, What does it do?

fn gradient_descent_with_multiple_inputs_frozen_weights() {
    let mut weights: Vector = vec![0.1, 0.2, -0.1];

    let toes = vec![8.5, 9.5, 9.9, 9.0];
    let wlrec = vec![0.65, 0.8, 0.8, 0.9];
    let nfans = vec![1.2, 1.3, 0.5, 1.0];

    let input = vec![toes[0], wlrec[0], nfans[0]];

    let win_or_lose_binary = [1.0, 1.0, 0.0, 1.0];
    let truth = win_or_lose_binary[0];

    let alpha = 0.3;

    for i in 0..3 {
        println!("Iteration {}", i + 1);

        let pred = neural_network_1(input.clone(), &weights);
        let error = (pred - truth).powf(2.0);
        println!("Error: {}, Prediction: {}", error, pred);

        let delta = pred - truth;
        let mut weight_delta = elementwise_scalar_multiplication(input.clone(), delta);
        weight_delta[0] = 0.0;

        for i in 0..3 {
            weights[i] -= alpha * weight_delta[i];
        }
        println!(
            "Weights: {:?}, Weight Deltas: {:?}\n",
            weights, weight_delta
        );
    }
}

/// Gradient Descent Learning with multiple outputs

fn gradient_descent_with_multiple_outputs() {
    let mut weights = vec![0.3, 0.2, 0.9];

    let wlrec = vec![0.65, 1.0, 1.0, 0.9];

    let hurt = vec![0.1, 0.0, 0.0, 0.1];
    let win = vec![1.0, 1.0, 0.0, 1.0];
    let sad = vec![0.1, 0.0, 0.1, 0.2];

    let input = wlrec[0];
    let truth = vec![hurt[0], win[0], sad[0]];

    let alpha = 0.1;

    let pred = neural_network_2(input, weights.clone());
    let error: Vector = pred
        .iter()
        .zip(truth.iter())
        .map(|(x, y)| (x - y).powf(2.0))
        .collect();
    println!("Prediction: {:?}, Error: {:?}", pred, error);

    let deltas: Vector = pred.iter().zip(truth.iter()).map(|(x, y)| x - y).collect();

    // NOTE: mistake in book.
    let weight_deltas: Vector = elementwise_scalar_multiplication(deltas, input);

    for i in 0..weight_deltas.len() {
        weights[i] -= weight_deltas[i] * alpha;
    }

    println!("Weights: {:?}, Weight Deltas: {:?}", weights, weight_deltas);
}

fn neural_network_2(input: f64, weights: Vector) -> Vector {
    elementwise_scalar_multiplication(weights, input)
}

/// Gradient Descent with multiple inputs and outputs

fn gradient_descent_with_multiple_inputs_and_outputs() {
    let toes = vec![8.5, 9.5, 9.9, 9.0];
    let wlrec = vec![0.65, 0.8, 0.8, 0.9];
    let nfans = vec![1.2, 1.3, 0.5, 1.0];

    let hurt = vec![0.1, 0.0, 0.0, 0.1];
    let win = vec![1.0, 1.0, 0.0, 1.0];
    let sad = vec![0.1, 0.0, 0.1, 0.2];

    let inputs = vec![toes[0], wlrec[0], nfans[0]];
    let mut weights = vec![
        vec![0.1, 0.1, -0.3],
        vec![0.1, 0.2, 0.0],
        vec![0.0, 1.3, 0.1],
    ];
    let truth = vec![hurt[0], win[0], sad[0]];

    let alpha = 0.01;

    let pred = neural_network_3(&inputs, weights.clone());
    let errors: Vector = pred
        .iter()
        .zip(truth.iter())
        .map(|(x, y)| (x - y).powf(2.0))
        .collect();

    println!("Prediction: {:?}, Error: {:?}", pred, errors);

    let deltas: Vector = pred.iter().zip(truth.iter()).map(|(p, t)| p - t).collect();
    let weight_deltas: Matrix = deltas
        .iter()
        .map(|i| elementwise_scalar_multiplication(inputs.clone(), *i))
        .collect();

    for i in 0..weights.len() {
        for j in 0..weights[i].len() {
            weights[i][j] -= alpha * weight_deltas[i][j];
        }
    }

    // NOTE: the saved weights output in the notebook is wrong.
    println!("Weights: {:?}, Weight Deltas: {:?}", weights, weight_deltas);
}

fn neural_network_3(inputs: &Vector, weights: Matrix) -> Vector {
    matrix_vector_multiplication(weights, inputs)
}
