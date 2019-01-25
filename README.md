# Grokking Deep Learning Rust

The exercises from the @iamtrask book [Grokking Deep Learning](https://manning.com/books/grokking-deep-learning) implemented in rust.

This crate isn't published, because ideally you'd do this on your own, but if you insist

```
cargo add grokking_deep_learning_rs --git https://github.com/suyash/grokking-deep-learning-rs
```

This crate is structured as a library, with the core library describing some common primitives used throughout and the individual chapters implemented in the exercises. To run the exercises from a particular chapter, for example chapter 12

```
cargo run --example chapter12
```

Currently this uses [rulinalg](https://docs.rs/rulinalg) for matrix operations, which uses a Rust implementation of `dgemm` and provides a 3x performance over normal ijk multiplication (see included benchmark). However, it still isn't as fast as numpy because it isn't multi-threaded. Currently working on something of my own.

The __datasets__ are extracted into a [separate library crate](https://github.com/suyash/datasets), which currently provides functions for loading 4 datasets, and an iterator for batching and shuffling. Planning to add more. Can be added using

```
cargo add datasets --git https://github.com/suyash/datasets
```

As a result of slower matmul, chapter 8 onwards, certain examples are smaller in size compared to the python examples.

The Chapter 13 core components were extracted into the core library, so they could be used in later chapters.

So, something like

```rust
use rulinalg::matrix::Matrix;

use grokking_deep_learning_rs::activations::{Sigmoid, Tanh};
use grokking_deep_learning_rs::layers::{Layer, Linear, Sequential};
use grokking_deep_learning_rs::losses::{Loss, MSELoss};
use grokking_deep_learning_rs::optimizers::{Optimizer, SGDOptimizer};
use grokking_deep_learning_rs::tensor::Tensor;

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
```

In Chapter 14, the RNN and LSTM examples have vanishing gradients and loss keeps going to NaN. There seems to be some kind of logic bomb in the code, where something is not doing what I think it does, still investigating.

# License

This project is licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
   http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or
   http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in Grokking-Deep-Learning-Rust by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
