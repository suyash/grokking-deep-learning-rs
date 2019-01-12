# Grokking Deep Learning Rust

The exercises from the @iamtrask book [Grokking Deep Learning](https://manning.com/books/grokking-deep-learning) implemented in rust.

Currently this uses [rulinalg](https://docs.rs/rulinalg) for matrix operations, which uses a Rust implementation of `dgemm` and provides a 3x performance over normal ijk multiplication (see included benchmark). However, it still isn't as fast as numpy because it isn't multi-threaded. Currently working on something of my own.

As a result of slower matmul, chapter 8 onwards, certain examples are smaller in size compared to the python examples.

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
