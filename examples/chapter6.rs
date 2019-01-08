use rand::distributions::Standard;

use grokking_deep_learning_rs::{
    dot, matrix_matrix_dot, random_matrix, relu_matrix, relu_vector, relu_vector_derivative,
    vector_matrix_dot, vector_vector_multiplication, Matrix,
};

fn main() {
    println!("\nCreating a Matrix or Two in Python\n");
    creating_a_matrix_or_two();

    println!("\nLearning the whole dataset!\n");
    learning_the_whole_dataset();

    println!("\nOur First \"Deep\" Neural Network\n");
    first_deep_neural_network();

    println!("\nBackpropagation\n");
    backpropagation();
}

/// Creating a Matrix or Two

fn creating_a_matrix_or_two() {
    let streetlights = vec![
        vec![1.0, 0.0, 1.0],
        vec![0.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0],
        vec![1.0, 1.0, 1.0],
        vec![0.0, 1.0, 1.0],
        vec![1.0, 0.0, 1.0],
    ];

    let walk_vs_stop = vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0];

    let mut weights = vec![0.5, 0.48, -0.7];

    let input = &streetlights[0];
    let goal_prediction = walk_vs_stop[0];

    let alpha = 0.1;

    for _ in 0..20 {
        let prediction = dot(input, &weights);
        let error = (goal_prediction - prediction).powi(2);
        println!("Prediction: {}, Error: {}", prediction, error);

        let delta = prediction - goal_prediction;
        for i in 0..3 {
            weights[i] = weights[i] - alpha * (input[i] * delta);
        }
    }
}

/// Learning the whole dataset!

fn learning_the_whole_dataset() {
    let streetlights = vec![
        vec![1.0, 0.0, 1.0],
        vec![0.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0],
        vec![1.0, 1.0, 1.0],
        vec![0.0, 1.0, 1.0],
        vec![1.0, 0.0, 1.0],
    ];

    let walk_vs_stop = vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0];

    let mut weights = vec![0.5, 0.48, -0.7];

    let alpha = 0.1;

    for i in 0..40 {
        let mut total_error = 0.0;

        for r in 0..streetlights.len() {
            let input = &streetlights[r];
            let goal_prediction = walk_vs_stop[r];

            let prediction = dot(input, &weights);
            println!("Prediction: {}", prediction);

            let error = (goal_prediction - prediction).powi(2);

            total_error += error;

            let delta = prediction - goal_prediction;
            for i in 0..3 {
                weights[i] = weights[i] - alpha * (input[i] * delta);
            }
        }

        println!("Error after iteration {} = {}\n", i + 1, total_error);
    }

    println!("Learned Weights: {:?}", weights);
}

/// Our first "Deep" Neural Network

fn first_deep_neural_network() {
    let inputs = vec![
        vec![1.0, 0.0, 1.0],
        vec![0.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0],
        vec![1.0, 1.0, 1.0],
    ];

    let outputs = vec![vec![1.0], vec![1.0], vec![0.0], vec![0.0]];

    let (alpha, hidden_size) = (0.2, 4);

    let mut weights_1: Matrix = random_matrix(3, hidden_size, &Standard);
    let mut weights_2: Matrix = random_matrix(hidden_size, 1, &Standard);

    let hidden_layer = relu_matrix(&matrix_matrix_dot(&inputs, &weights_1));
    let output = matrix_matrix_dot(&hidden_layer, &weights_2);
}

/// Backpropagation

fn backpropagation() {
    let inputs = vec![
        vec![1.0, 0.0, 1.0],
        vec![0.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0],
        vec![1.0, 1.0, 1.0],
    ];

    let outputs = vec![vec![1.0], vec![1.0], vec![0.0], vec![0.0]];

    let alpha = 0.2;

    // Weight values taken from the python notebooks for reproducing results.
    let mut weights_0_1: Matrix = vec![
        vec![-0.16595599, 0.44064899, -0.99977125, -0.39533485],
        vec![-0.70648822, -0.81532281, -0.62747958, -0.30887855],
        vec![-0.20646505, 0.07763347, -0.16161097, 0.370439],
    ];

    let mut weights_1_2: Matrix = vec![
        vec![-0.5910955],
        vec![0.75623487],
        vec![-0.94522481],
        vec![0.34093502],
    ];

    for it in 0..60 {
        let mut total_error = 0.0;

        for i in 0..4 {
            let hidden_layer = relu_vector(&vector_matrix_dot(&inputs[i], &weights_0_1));
            let prediction = vector_matrix_dot(&hidden_layer, &weights_1_2)[0];

            let error: f64 = (prediction - outputs[i][0]).powi(2);
            total_error += error;

            let delta_2_1 = prediction - outputs[i][0];
            let delta_1_0 = vector_vector_multiplication(
                &weights_1_2.iter().map(|v| v[0] * delta_2_1).collect(),
                &relu_vector_derivative(&hidden_layer),
            );

            let weight_deltas_1_2: Matrix =
                hidden_layer.iter().map(|v| vec![v * delta_2_1]).collect();

            let weight_deltas_0_1: Matrix = inputs[i]
                .iter()
                .map(|v| delta_1_0.iter().map(|v2| v * v2).collect())
                .collect();

            for i in 0..weights_1_2.len() {
                for j in 0..weights_1_2[i].len() {
                    weights_1_2[i][j] -= alpha * weight_deltas_1_2[i][j];
                }
            }

            for i in 0..weights_0_1.len() {
                for j in 0..weights_0_1[i].len() {
                    weights_0_1[i][j] -= alpha * weight_deltas_0_1[i][j];
                }
            }
        }

        if (it + 1) % 10 == 0 {
            println!("Error: {}", total_error);
        }
    }
}
