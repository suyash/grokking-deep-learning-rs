#![feature(test)]

extern crate test;

use test::{black_box, Bencher};

use std::ops::Mul;

use rulinalg::matrix::Matrix;

use grokking_deep_learning_rs::matrix_matrix_dot;

#[bench]
fn bench_normal(b: &mut Bencher) {
    b.iter(|| {
        let m1 = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let m2 = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        black_box(matrix_matrix_dot(&m1, &m2));
    });
}

#[bench]
fn bench_rulinalg(b: &mut Bencher) {
    b.iter(|| {
        let m1 = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let m2 = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        black_box(m1.mul(m2));
    });
}
