//! This was extracted from the Chapter 13 exercises and moved into the core library so it could be used in later chapters.

use std::iter::FromIterator;
use std::fmt;

use rand::distributions::Uniform;
use rulinalg::matrix::{BaseMatrix, Matrix};
use std::rc::Rc;

use crate::generate_random_vector;
use crate::tensor::{Dot, Expand, Tensor};

pub trait Layer {
    fn forward(&self, inputs: &[&Tensor]) -> Vec<Tensor>;

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }
}

#[derive(Debug)]
pub struct Linear {
    weights: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(n_inputs: usize, n_outputs: usize, bias: bool) -> Linear {
        let distribution = Uniform::new(0.0, 1.0);

        let weights = Tensor::new_const(Matrix::new(
            n_inputs,
            n_outputs,
            generate_random_vector(n_inputs * n_outputs, 0.5, 0.0, &distribution),
        ));

        let bias = if bias {
            Some(Tensor::new_const(Matrix::zeros(1, n_outputs)))
        } else {
            None
        };

        Linear { weights, bias }
    }
}

impl Layer for Linear {
    fn forward(&self, inputs: &[&Tensor]) -> Vec<Tensor> {
        let rows = inputs[0].0.borrow().data.rows();
        match &self.bias {
            None => vec![inputs[0].dot(&self.weights)],
            Some(bias) => vec![&inputs[0].dot(&self.weights) + &bias.expand(0, rows)],
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        match &self.bias {
            None => vec![&self.weights],
            Some(bias) => vec![&self.weights, bias],
        }
    }
}

pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
}

impl fmt::Debug for Sequential {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Sequential {{  }}")
    }
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        Sequential { layers }
    }

    #[allow(dead_code)]
    fn add(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }
}

impl Layer for Sequential {
    fn forward(&self, inputs: &[&Tensor]) -> Vec<Tensor> {
        // TODO: can this be avoided
        let mut input = Tensor(Rc::clone(&inputs[0].0));

        for layer in self.layers.iter() {
            input = layer.forward(&[&input]).remove(0);
        }

        vec![input]
    }

    fn parameters(&self) -> Vec<&Tensor> {
        self.layers
            .iter()
            .map(|l| l.parameters())
            .flat_map(|v| v.into_iter())
            .collect()
    }
}

#[derive(Debug)]
pub struct Embedding {
    weights: Tensor,
}

impl Embedding {
    pub fn new(vocab_size: usize, embedding_size: usize) -> Embedding {
        let distribution = Uniform::new(0.0, 1.0);
        Embedding {
            weights: Tensor::new_const(Matrix::new(
                vocab_size,
                embedding_size,
                generate_random_vector(
                    vocab_size * embedding_size,
                    1.0 / (embedding_size as f64),
                    -0.5 / (embedding_size as f64),
                    &distribution,
                ),
            )),
        }
    }
}

impl Layer for Embedding {
    fn forward(&self, inputs: &[&Tensor]) -> Vec<Tensor> {
        let data = Vec::from_iter(
            inputs[0]
                .0
                .borrow()
                .data
                .row(0)
                .raw_slice()
                .iter()
                .map(|v| (*v as usize)),
        );

        vec![self.weights.index_select(data)]
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weights]
    }
}

pub struct RNNCell {
    n_hidden: usize,
    w_ih: Linear,
    w_hh: Linear,
    w_ho: Linear,
    activation: Box<dyn Layer>,
}

impl fmt::Debug for RNNCell {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "RNNCell {{ n_hidden: {:?}, w_ih: {:?}, w_hh: {:?}, w_ho: {:?} }}", self.n_hidden, self.w_ih, self.w_hh, self.w_ho)
    }
}

impl RNNCell {
    pub fn new(
        n_inputs: usize,
        n_hidden: usize,
        n_outputs: usize,
        activation: Box<dyn Layer>,
    ) -> RNNCell {
        let w_ih = Linear::new(n_inputs, n_hidden, true);
        let w_hh = Linear::new(n_hidden, n_hidden, true);
        let w_ho = Linear::new(n_hidden, n_outputs, true);

        RNNCell {
            n_hidden,
            w_ih,
            w_hh,
            w_ho,
            activation,
        }
    }

    pub fn create_start_state(&self, batch_size: usize) -> Tensor {
        Tensor::new_const(Matrix::zeros(batch_size, self.n_hidden))
    }
}

impl Layer for RNNCell {
    fn forward(&self, inputs: &[&Tensor]) -> Vec<Tensor> {
        let (input, hidden) = (inputs[0], inputs[1]);

        let state_part = self.w_hh.forward(&[hidden]);
        let input_part = self.w_ih.forward(&[input]);

        let mut new_state = self
            .activation
            .forward(&[&(&input_part[0] + &state_part[0])]);
        let mut output = self.w_ho.forward(&[&new_state[0]]);

        vec![output.remove(0), new_state.remove(0)]
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut ans = self.w_ih.parameters();
        ans.append(&mut self.w_hh.parameters());
        ans.append(&mut self.w_ho.parameters());
        ans
    }
}
