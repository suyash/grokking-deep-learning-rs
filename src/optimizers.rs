//! This was extracted from the Chapter 13 exercises and moved into the core library so it could be used in later chapters.

use rulinalg::matrix::BaseMatrix;

use crate::tensor::Tensor;

pub trait Optimizer {
    fn step(&self, zero: bool);
}

#[derive(Debug)]
pub struct SGDOptimizer<'a> {
    parameters: Vec<&'a Tensor>,
    alpha: f64,
}

impl<'a> SGDOptimizer<'a> {
    pub fn new(parameters: Vec<&'a Tensor>, alpha: f64) -> SGDOptimizer {
        SGDOptimizer { parameters, alpha }
    }

    fn step_parameter(&self, parameter: &'a Tensor, zero: bool) {
        let mut w = parameter.0.borrow_mut();
        let grad = w.grad.take();

        if zero {
            w.grad = None;
        }

        let grad = grad.unwrap();
        let grad = &grad.borrow().data;

        for i in 0..w.data.rows() {
            for j in 0..w.data.cols() {
                w.data[[i, j]] -= self.alpha * grad[[i, j]];
            }
        }
    }
}

impl<'a> Optimizer for SGDOptimizer<'a> {
    fn step(&self, zero: bool) {
        for p in self.parameters.iter() {
            self.step_parameter(p, zero);
        }
    }
}
