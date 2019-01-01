fn main() {
    println!("\nLearning using hot and cold method\n");
    hot_and_cold_method();

    println!("\nHot and Cold Learning\n");
    hot_and_cold_learning();

    println!("\nCalculating both direction and amount from error.\n");
    hot_and_cold_learning_with_direction_and_amount();

    println!("\nOne Iteration of Gradient Descent\n");
    gradient_descent_method();

    println!("\nLearning is just reducing error\n");
    gradient_descent();

    println!("\nLet's watch several steps of learning\n");
    gradient_descent_2();

    println!("\nWhy does this work? What really is weight delta?\n");
    gradient_descent_3();

    println!("\nBreaking Gradient Descent\n");
    gradient_descent_working();
    println!();
    gradient_descent_breaking();

    println!("\nAlpha\n");
    gradient_descent_working_again();
}

/// Learning using hot and cold method

fn hot_and_cold_method() {
    let (mut weight, lr) = (0.1, 0.01);
    let (number_of_toes, win_or_lose_binary) = ([8.5], [1.0]);

    let (input, truth) = (number_of_toes[0], win_or_lose_binary[0]);

    let pred = neural_network(input, weight);

    let err = (pred - truth).powf(2.0);
    println!("error: {}", err);

    let (pred_up, pred_down) = (
        neural_network(input, weight + lr),
        neural_network(input, weight - lr),
    );
    let (err_up, err_down) = ((pred_up - truth).powf(2.0), (pred_down - truth).powf(2.0));
    println!("error up: {}, error down: {}", err_up, err_down);

    if err_up < err_down {
        weight += lr;
    } else {
        weight -= lr;
    }
}

/// Hot and Cold Learning

fn hot_and_cold_learning() {
    let mut weight = 0.5;

    let (input, truth) = (0.5, 0.8);

    let n_iterations = 20;
    let lr = 0.001;

    for _ in 0..n_iterations {
        let pred = neural_network(input, weight);

        let err = (pred - truth).powf(2.0);
        println!("Error: {}, Prediction: {}", err, pred);

        let (pred_up, pred_down) = (
            neural_network(input, weight + lr),
            neural_network(input, weight - lr),
        );
        let (err_up, err_down) = ((pred_up - truth).powf(2.0), (pred_down - truth).powf(2.0));

        if err_up < err_down {
            weight += lr;
        } else if err_up > err_down {
            weight -= lr;
        }
    }
}

/// Calculating both direction and amount from error.

fn hot_and_cold_learning_with_direction_and_amount() {
    let mut weight = 0.5;

    let (input, truth) = (0.5, 0.8);

    let n_iterations = 1101;

    for _ in 0..n_iterations {
        let pred = neural_network(input, weight);

        let err = (pred - truth).powf(2.0);
        println!("Error: {}, Prediction: {}", err, pred);

        let direction_and_amount = (pred - truth) * input;
        weight -= direction_and_amount;
    }
}

/// One Iteration of Gradient Descent

fn gradient_descent_method() {
    let (mut weight, alpha) = (0.1, 0.01);
    let (number_of_toes, win_or_lose_binary) = ([8.5], [1.0]);

    let (input, truth) = (number_of_toes[0], win_or_lose_binary[0]);

    let pred = neural_network(input, truth);
    let err = (pred - truth).powf(2.0);

    let delta = pred - truth;
    let weight_delta = input * delta;

    let alpha = 0.01;
    weight -= weight_delta * alpha;
}

fn neural_network(input: f64, weight: f64) -> f64 {
    input * weight
}

/// Learning is just reducing error

fn gradient_descent() {
    let (mut weight, truth, input) = (0.0, 0.8, 0.5);
    for _ in 0..4 {
        let pred = neural_network(input, weight);
        let err = (pred - truth).powf(2.0);
        println!("Error: {}, Prediction: {}", err, pred);

        let delta = pred - truth;
        let weight_delta = delta * input;
        weight -= weight_delta;
    }
}

/// Let's watch several steps of learning.

fn gradient_descent_2() {
    let (mut weight, truth, input) = (0.0, 0.8, 1.1);
    for _ in 0..4 {
        println!("------\nWeight: {}", weight);

        let pred = neural_network(input, weight);
        let err = (pred - truth).powf(2.0);
        println!("Error: {}, Prediction: {}", err, pred);

        let delta = pred - truth;
        let weight_delta = delta * input;
        weight -= weight_delta;
        println!("Delta: {}, Weight Delta: {}", delta, weight_delta);
    }
}

/// Why does this work? What really is weight delta?

fn gradient_descent_3() {
    let (mut weight, truth, input) = (0.0, 0.8, 1.1);
    for _ in 0..20 {
        let pred = neural_network(input, weight);
        let err = (pred - truth).powf(2.0);
        println!("Error: {}, Prediction: {}", err, pred);

        let delta = pred - truth;
        let weight_delta = delta * input;
        weight -= weight_delta;
    }
}

/// Breaking Gradient Descent

fn gradient_descent_working() {
    let (mut weight, truth, input) = (0.5, 0.8, 0.5);
    for _ in 0..20 {
        let pred = neural_network(input, weight);
        let err = (pred - truth).powf(2.0);
        println!("Error: {}, Prediction: {}", err, pred);

        let delta = pred - truth;
        let weight_delta = delta * input;
        weight -= weight_delta;
    }
}

fn gradient_descent_breaking() {
    let (mut weight, truth, input) = (0.5, 0.8, 2.0);
    for _ in 0..20 {
        let pred = neural_network(input, weight);
        let err = (pred - truth).powf(2.0);
        println!("Error: {}, Prediction: {}", err, pred);

        let delta = pred - truth;
        let weight_delta = delta * input;
        weight -= weight_delta;
    }
}

/// Alpha

fn gradient_descent_working_again() {
    let (mut weight, truth, input) = (0.5, 0.8, 2.0);
    let alpha = 0.1;

    for _ in 0..20 {
        let pred = neural_network(input, weight);
        let err = (pred - truth).powf(2.0);
        println!("Error: {}, Prediction: {}", err, pred);

        let delta = pred - truth;
        let weight_delta = delta * input;
        weight -= alpha * weight_delta;
    }
}
