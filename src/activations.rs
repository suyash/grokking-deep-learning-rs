//! This was extracted from the Chapter 13 exercises and moved into the core library so it could be used in later chapters.

use crate::layers::Layer;
use crate::tensor::Tensor;

pub struct Sigmoid;

impl Layer for Sigmoid {
    fn forward(&self, inputs: &[&Tensor]) -> Vec<Tensor> {
        vec![inputs[0].sigmoid()]
    }
}

pub struct Tanh;

impl Layer for Tanh {
    fn forward(&self, inputs: &[&Tensor]) -> Vec<Tensor> {
        vec![inputs[0].tanh()]
    }
}

pub struct Relu;

impl Layer for Relu {
    fn forward(&self, inputs: &[&Tensor]) -> Vec<Tensor> {
        vec![inputs[0].relu()]
    }
}
