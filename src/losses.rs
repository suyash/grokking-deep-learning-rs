//! This was extracted from the Chapter 13 exercises and moved into the core library so it could be used in later chapters.

use crate::tensor::{Sum, Tensor};

pub trait Loss {
    fn forward(&self, pred: &Tensor, target: &Tensor) -> Tensor;
}

pub struct MSELoss;

impl Loss for MSELoss {
    fn forward(&self, pred: &Tensor, target: &Tensor) -> Tensor {
        (&(pred - target) * &(pred - target)).sum(0)
    }
}

pub struct CrossEntropyLoss;

impl Loss for CrossEntropyLoss {
    fn forward(&self, pred: &Tensor, target_indices: &Tensor) -> Tensor {
        pred.cross_entropy(target_indices)
    }
}
