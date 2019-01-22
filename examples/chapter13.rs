//! Chapter13 - Intro to Automatic Differentiation - Let's Build A Deep Learning Framework
//!
//! https://github.com/iamtrask/Grokking-Deep-Learning/blob/master/Chapter13%20-%20Intro%20to%20Automatic%20Differentiation%20-%20Let's%20Build%20A%20Deep%20Learning%20Framework.ipynb

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::iter::FromIterator;
use std::ops::{Add, Mul, Neg, Sub};
use std::rc::Rc;

use datasets::text::babi_en_single_supporting_fact_task;
use datasets::Dataset;
use rand::distributions::Uniform;
use rand::{thread_rng, RngCore};
use rulinalg::matrix::{BaseMatrix, Matrix};

use grokking_deep_learning_rs::{argmax, generate_random_vector};

fn main() {
    println!("\nIntroduction to Tensors\n");
    introduction_to_tensors();

    println!("\nIntroduction to Autograd\n");
    introduction_to_autograd();
    introduction_to_autograd_2();

    println!("\nAutograd with multiple tensors\n");
    autograd_with_multiple_tensors();
    autograd_neg();

    println!("\nUsing Autograd ot train a Neural Network\n");
    training_using_autograd();

    println!("\nAdding Automatic Optimization\n");
    training_with_automatic_optimization();

    println!("\nLayers Which Contain Layers\n");
    layers_which_contain_layers();

    println!("\nLoss Function Layers\n");
    loss_function_layers();

    println!("\nNonLinearity Layers\n");
    nonlinearity_layers();

    println!("\nEmbedding Layers\n");
    embedding_layer();

    println!("\nThe Embedding Layer\n");
    cross_entropy_loss();

    println!("\nRecurrent Neural Network\n");
    recurrent_neural_network().unwrap();
}

fn introduction_to_tensors() {
    let t1 = BasicTensor1 { data: vec![0.0] };
    let t2 = BasicTensor1 { data: vec![1.0] };
    println!("{:?}", t1 + t2);
}

#[derive(Debug)]
struct BasicTensor1 {
    data: Vec<f64>,
}

impl Add for BasicTensor1 {
    type Output = BasicTensor1;

    fn add(self, other: BasicTensor1) -> Self::Output {
        BasicTensor1 {
            data: self
                .data
                .into_iter()
                .zip(other.data.into_iter())
                .map(|(a, b)| a + b)
                .collect(),
        }
    }
}

fn introduction_to_autograd() {
    let x = BasicTensor2::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = BasicTensor2::new(vec![2.0; 5]);

    let mut z = x + y;
    println!("{:?}", z);

    z.backward(BasicTensor2::new(vec![1.0, 1.0, 1.0, 1.0, 1.0]));

    let xy = z.creators.unwrap();

    println!("{:?}", xy[0].grad);
    println!("{:?}", xy[1].grad);
}

#[derive(Debug, Clone)]
enum BasicOperation {
    Add,
    Const,
}

#[derive(Debug, Clone)]
struct BasicTensor2 {
    data: Vec<f64>,
    grad: Option<Box<BasicTensor2>>,
    creation_op: BasicOperation,
    creators: Option<Vec<BasicTensor2>>,
}

impl BasicTensor2 {
    fn new(data: Vec<f64>) -> Self {
        BasicTensor2 {
            data,
            grad: None,
            creation_op: BasicOperation::Const,
            creators: None,
        }
    }

    fn backward(&mut self, grad: BasicTensor2) {
        match self.creation_op {
            BasicOperation::Add => {
                for c in self.creators.as_mut().unwrap().iter_mut() {
                    c.backward(grad.clone());
                }
            }
            _ => {
                self.grad = Some(Box::new(grad));
            }
        };
    }
}

impl Add for BasicTensor2 {
    type Output = BasicTensor2;

    fn add(self, other: Self) -> BasicTensor2 {
        BasicTensor2 {
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a + b)
                .collect(),
            grad: None,
            creation_op: BasicOperation::Add,
            creators: Some(vec![self, other]),
        }
    }
}

#[allow(clippy::many_single_char_names)]
fn introduction_to_autograd_2() {
    let a = BasicTensor2::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let b = BasicTensor2::new(vec![2.0; 5]);
    let c = BasicTensor2::new(vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    let d = BasicTensor2::new(vec![-1.0, -2.0, -3.0, -4.0, -5.0]);

    let e = a + b;
    let f = c + d;
    let mut g = e + f;

    g.backward(BasicTensor2::new(vec![1.0, 1.0, 1.0, 1.0, 1.0]));
    println!("{:?}", g);

    let ef = g.creators.as_ref().unwrap();
    let ab = ef[0].creators.as_ref().unwrap();

    let a = &ab[0];
    println!("{:?}", a.grad);
}

#[allow(clippy::many_single_char_names)]
fn autograd_with_multiple_tensors() {
    let a = Tensor::new_const(Matrix::new(1, 5, vec![1.0, 2.0, 3.0, 4.0, 5.0]));
    let b = Tensor::new_const(Matrix::new(1, 5, vec![2.0, 2.0, 2.0, 2.0, 2.0]));
    let c = Tensor::new_const(Matrix::new(1, 5, vec![5.0, 4.0, 3.0, 2.0, 1.0]));

    let d = &a + &b;
    let e = &b + &c;
    let f = &d + &e;

    // println!("{:#?}", f);
    f.backward(Tensor::grad(Matrix::new(
        1,
        5,
        vec![1.0, 1.0, 1.0, 1.0, 1.0],
    )));
    println!("{:?}", b.0.borrow().grad);
}

#[allow(clippy::many_single_char_names)]
fn autograd_neg() {
    let a = Tensor::new_const(Matrix::new(1, 5, vec![1.0, 2.0, 3.0, 4.0, 5.0]));
    let b = Tensor::new_const(Matrix::new(1, 5, vec![2.0, 2.0, 2.0, 2.0, 2.0]));
    let c = Tensor::new_const(Matrix::new(1, 5, vec![5.0, 4.0, 3.0, 2.0, 1.0]));

    let d = &a + &(-&b);
    let e = &(-&b) + &c;
    let f = &d + &e;

    f.backward(Tensor::grad(Matrix::new(
        1,
        5,
        vec![1.0, 1.0, 1.0, 1.0, 1.0],
    )));
    println!("{:?}", b.0.borrow().grad);
}

#[derive(Debug, Clone)]
enum Operation {
    Const,
    Add,
    Neg,
    Sub,
    Mul,
    Dot,
    Transpose,
    Sigmoid,
    Tanh,
    Relu,
    Sum(usize),
    Expand(usize),
    // This is not generic as implemented for python
    // and can only select indices on the 0th axis. Hence, only a vector.
    IndexSelect(Vec<usize>),
    CrossEntropy(Matrix<f64>, Matrix<f64>),
}

type TensorRef = Rc<RefCell<TensorImpl>>;

#[derive(Debug)]
struct TensorImpl {
    id: u64,
    data: Matrix<f64>,
    grad: Option<TensorRef>,
    creation_op: Operation,
    creators: Option<Vec<TensorRef>>,
    autograd: bool,
    children: BTreeMap<u64, usize>,
}

impl TensorImpl {
    fn grad(data: Matrix<f64>) -> Self {
        TensorImpl {
            id: thread_rng().next_u64(),
            data,
            grad: None,
            creation_op: Operation::Const,
            creators: None,
            autograd: false,
            children: BTreeMap::new(),
        }
    }

    fn all_children_grads_accounted_for(&self) -> bool {
        self.children.iter().all(|(_, c)| c == &0)
    }

    #[allow(clippy::cyclomatic_complexity)]
    fn backward(&mut self, grad: TensorRef, grad_from: Option<u64>) {
        if self.autograd {
            if let Some(grad_from) = &grad_from {
                if self.children[&grad_from] == 0 {
                    panic!("Can only Backpropagate through a tensor once");
                } else {
                    self.children
                        .insert(*grad_from, self.children[grad_from] - 1);
                }
            }

            self.grad = match self.grad.take() {
                None => Some(Rc::new(RefCell::new(TensorImpl::grad(
                    grad.borrow().data.clone(),
                )))),
                Some(current_grad) => {
                    {
                        let current_grad_data = &mut current_grad.borrow_mut().data;
                        let current_grad_raw = current_grad_data.mut_data();
                        let grad_data = &grad.borrow().data;
                        let grad = grad_data.data();
                        for i in 0..grad.len() {
                            current_grad_raw[i] += grad[i];
                        }
                    }

                    Some(current_grad)
                }
            };

            if self.creators.is_some()
                && (self.all_children_grads_accounted_for() || grad_from.is_none())
            {
                let grad = self.grad.as_ref().unwrap();
                let creators = self.creators.as_ref().unwrap();

                match &self.creation_op {
                    Operation::Add => {
                        creators[0]
                            .borrow_mut()
                            .backward(Rc::clone(grad), Some(self.id));
                        creators[1]
                            .borrow_mut()
                            .backward(Rc::clone(grad), Some(self.id));
                    }
                    Operation::Neg => {
                        let data = &grad.borrow().data;
                        let data_data: Vec<f64> = data.data().iter().map(|v| -v).collect();
                        creators[0].borrow_mut().backward(
                            Rc::new(RefCell::new(TensorImpl::grad(Matrix::new(
                                data.rows(),
                                data.cols(),
                                data_data,
                            )))),
                            Some(self.id),
                        );
                    }
                    Operation::Sub => {
                        creators[0]
                            .borrow_mut()
                            .backward(Rc::clone(grad), Some(self.id));
                        {
                            let data = &grad.borrow().data;
                            let data_data: Vec<f64> = data.data().iter().map(|v| -v).collect();
                            creators[1].borrow_mut().backward(
                                Rc::new(RefCell::new(TensorImpl::grad(Matrix::new(
                                    data.rows(),
                                    data.cols(),
                                    data_data,
                                )))),
                                Some(self.id),
                            );
                        }
                    }
                    Operation::Mul => {
                        let grad = &grad.borrow().data;

                        let grad0 = {
                            let grad0 = &creators[1].borrow().data;
                            let grad0_data: Vec<f64> = grad0
                                .data()
                                .iter()
                                .zip(grad.data().iter())
                                .map(|(a, b)| a * b)
                                .collect();
                            let grad0 = Matrix::new(grad0.rows(), grad0.cols(), grad0_data);
                            Rc::new(RefCell::new(TensorImpl::grad(grad0)))
                        };

                        let grad1 = {
                            let grad1 = &creators[0].borrow().data;
                            let grad1_data: Vec<f64> = grad1
                                .data()
                                .iter()
                                .zip(grad.data().iter())
                                .map(|(a, b)| a * b)
                                .collect();
                            let grad1 = Matrix::new(grad1.rows(), grad1.cols(), grad1_data);
                            Rc::new(RefCell::new(TensorImpl::grad(grad1)))
                        };

                        creators[0].borrow_mut().backward(grad0, Some(self.id));
                        creators[1].borrow_mut().backward(grad1, Some(self.id));
                    }
                    Operation::Transpose => {
                        let grad = &grad.borrow().data;
                        let data = grad.transpose();
                        creators[0]
                            .borrow_mut()
                            .backward(Rc::new(RefCell::new(TensorImpl::grad(data))), Some(self.id));
                    }
                    Operation::Dot => {
                        let grad = &grad.borrow().data;

                        let act_delta = {
                            let weights = &creators[1].borrow().data;
                            grad.mul(weights.transpose())
                        };

                        let weights_delta = {
                            let act = &creators[0].borrow().data;
                            act.transpose().mul(grad)
                        };

                        creators[0].borrow_mut().backward(
                            Rc::new(RefCell::new(TensorImpl::grad(act_delta))),
                            Some(self.id),
                        );

                        creators[1].borrow_mut().backward(
                            Rc::new(RefCell::new(TensorImpl::grad(weights_delta))),
                            Some(self.id),
                        );
                    }
                    Operation::Sum(axis) => {
                        let new_grad = {
                            let data = &creators[0].borrow().data;
                            let grad = &grad.borrow().data;
                            let mut new_grad = Matrix::zeros(data.rows(), data.cols());

                            if axis == &0 {
                                for i in 0..data.rows() {
                                    for j in 0..data.cols() {
                                        new_grad[[i, j]] = grad[[0, j]];
                                    }
                                }
                            } else if axis == &1 {
                                for i in 0..data.rows() {
                                    for j in 0..data.cols() {
                                        new_grad[[i, j]] = grad[[i, 0]];
                                    }
                                }
                            } else {
                                panic!("this is broken");
                            }

                            new_grad
                        };

                        creators[0].borrow_mut().backward(
                            Rc::new(RefCell::new(TensorImpl::grad(new_grad))),
                            Some(self.id),
                        );
                    }
                    Operation::Expand(dim) => {
                        let new_grad = {
                            let data = &creators[0].borrow().data;
                            let grad = &grad.borrow().data;
                            let mut new_grad = Matrix::zeros(data.rows(), data.cols());

                            if dim == &0 {
                                for i in 0..grad.rows() {
                                    for j in 0..grad.cols() {
                                        new_grad[[0, j]] += grad[[i, j]];
                                    }
                                }
                            } else {
                                panic!("this is broken");
                            }

                            new_grad
                        };

                        creators[0].borrow_mut().backward(
                            Rc::new(RefCell::new(TensorImpl::grad(new_grad))),
                            Some(self.id),
                        );
                    }
                    Operation::Sigmoid => {
                        let new_grad = {
                            let data = &creators[0].borrow().data;
                            let grad = &grad.borrow().data;

                            let mut new_grad = Matrix::zeros(grad.rows(), grad.cols());
                            for i in 0..grad.rows() {
                                for j in 0..grad.cols() {
                                    new_grad[[i, j]] =
                                        grad[[i, j]] * (data[[i, j]] * (1.0 - data[[i, j]]));
                                }
                            }

                            new_grad
                        };

                        creators[0].borrow_mut().backward(
                            Rc::new(RefCell::new(TensorImpl::grad(new_grad))),
                            Some(self.id),
                        );
                    }
                    Operation::Tanh => {
                        let new_grad = {
                            let data = &creators[0].borrow().data;
                            let grad = &grad.borrow().data;

                            let mut new_grad = Matrix::zeros(grad.rows(), grad.cols());
                            for i in 0..grad.rows() {
                                for j in 0..grad.cols() {
                                    new_grad[[i, j]] =
                                        grad[[i, j]] * (1.0 - (data[[i, j]] * data[[i, j]]));
                                }
                            }

                            new_grad
                        };

                        creators[0].borrow_mut().backward(
                            Rc::new(RefCell::new(TensorImpl::grad(new_grad))),
                            Some(self.id),
                        );
                    }
                    Operation::Relu => {
                        let new_grad = {
                            let data = &creators[0].borrow().data;
                            let grad = &grad.borrow().data;

                            let mut new_grad = Matrix::zeros(grad.rows(), grad.cols());
                            for i in 0..grad.rows() {
                                for j in 0..grad.cols() {
                                    new_grad[[i, j]] =
                                        grad[[i, j]] * if data[[i, j]] > 0.0 { 1.0 } else { 0.0 };
                                }
                            }

                            new_grad
                        };

                        creators[0].borrow_mut().backward(
                            Rc::new(RefCell::new(TensorImpl::grad(new_grad))),
                            Some(self.id),
                        );
                    }
                    Operation::IndexSelect(indices) => {
                        let new_grad = {
                            let data = &creators[0].borrow().data;
                            let grad = &grad.borrow().data;

                            let mut new_grad = Matrix::zeros(data.rows(), data.cols());
                            for (i, ix) in indices.iter().enumerate() {
                                for j in 0..data.cols() {
                                    new_grad[[*ix, j]] = grad[[i, j]];
                                }
                            }

                            new_grad
                        };

                        creators[0].borrow_mut().backward(
                            Rc::new(RefCell::new(TensorImpl::grad(new_grad))),
                            Some(self.id),
                        )
                    }
                    Operation::CrossEntropy(predictions, targets) => {
                        creators[0].borrow_mut().backward(
                            Rc::new(RefCell::new(TensorImpl::grad(predictions - targets))),
                            Some(self.id),
                        )
                    }
                    Operation::Const => {}
                }
            }
        }
    }
}

/// Tensor implements "shallow" clones, primarily so that they can be put inside enum variants.
#[derive(Debug)]
struct Tensor(TensorRef);

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Tensor(Rc::clone(&self.0))
    }
}

impl Tensor {
    fn new_const(data: Matrix<f64>) -> Self {
        Self::new(data, Operation::Const, None)
    }

    fn grad(data: Matrix<f64>) -> Self {
        let tensor_impl = TensorImpl::grad(data);
        Tensor(Rc::new(RefCell::new(tensor_impl)))
    }

    fn new(data: Matrix<f64>, creation_op: Operation, creators: Option<Vec<TensorRef>>) -> Self {
        let tensor_impl = TensorImpl {
            id: thread_rng().next_u64(),
            data,
            grad: None,
            creation_op,
            creators,
            autograd: true,
            children: BTreeMap::new(),
        };

        if let Some(creators) = &tensor_impl.creators {
            for c in creators.iter() {
                let children = &mut c.borrow_mut().children;
                let e = children.entry(tensor_impl.id).or_insert(0);
                *e += 1;
            }
        }

        Tensor(Rc::new(RefCell::new(tensor_impl)))
    }

    fn backward(&self, grad: Tensor) {
        self.0.borrow_mut().backward(grad.0, None);
    }

    /// higher order ops

    fn sigmoid(&self) -> Tensor {
        let result = {
            let data = &self.0.borrow().data;
            let mut ans = Matrix::zeros(data.rows(), data.cols());

            for i in 0..data.rows() {
                for j in 0..data.cols() {
                    ans[[i, j]] = 1.0 / (1.0 + (-data[[i, j]]).exp());
                }
            }

            ans
        };

        Tensor::new(result, Operation::Sigmoid, Some(vec![Rc::clone(&self.0)]))
    }

    fn tanh(&self) -> Tensor {
        let result = {
            let data = &self.0.borrow().data;
            let mut ans = Matrix::zeros(data.rows(), data.cols());

            for i in 0..data.rows() {
                for j in 0..data.cols() {
                    ans[[i, j]] = data[[i, j]].tanh();
                }
            }

            ans
        };

        Tensor::new(result, Operation::Tanh, Some(vec![Rc::clone(&self.0)]))
    }

    fn relu(&self) -> Tensor {
        let result = {
            let data = &self.0.borrow().data;
            let mut ans = Matrix::zeros(data.rows(), data.cols());

            for i in 0..data.rows() {
                for j in 0..data.cols() {
                    ans[[i, j]] = if data[[i, j]] > 0.0 {
                        data[[i, j]]
                    } else {
                        0.0
                    };
                }
            }

            ans
        };

        Tensor::new(result, Operation::Relu, Some(vec![Rc::clone(&self.0)]))
    }

    fn index_select(&self, indices: Vec<usize>) -> Tensor {
        let result = {
            let data = &self.0.borrow().data;
            let mut ans = Matrix::zeros(indices.len(), data.cols());

            for (i, ix) in indices.iter().enumerate() {
                for j in 0..data.cols() {
                    ans[[i, j]] = data[[*ix, j]];
                }
            }

            ans
        };

        Tensor::new(
            result,
            Operation::IndexSelect(indices),
            Some(vec![Rc::clone(&self.0)]),
        )
    }

    /// the current tensor and the targets have to be the same shape
    fn cross_entropy(&self, target_indices: &Tensor) -> Tensor {
        let (m, target_dist, loss) = {
            let data = &self.0.borrow().data;
            let target_indices = &target_indices.0.borrow().data;

            let mut rs = vec![0.0; data.rows()];

            let mut m = Matrix::zeros(data.rows(), data.cols());

            for i in 0..data.rows() {
                for j in 0..data.cols() {
                    m[[i, j]] = data[[i, j]].exp();
                    rs[i] += m[[i, j]];
                }
            }

            for i in 0..data.rows() {
                for j in 0..data.cols() {
                    m[[i, j]] /= rs[i];
                }
            }

            let mut target_dist = Matrix::zeros(data.rows(), data.cols());

            let mut loss = 0.0;
            for i in 0..target_indices.rows() {
                let index = target_indices[[i, 0]] as usize;
                target_dist[[i, index]] = 1.0;

                let current_loss = data[[i, index]].ln();
                loss += -current_loss;
            }

            loss /= -(data.rows() as f64);

            (m, target_dist, loss)
        };

        Tensor::new(
            Matrix::new(1, 1, vec![loss]),
            Operation::CrossEntropy(m, target_dist),
            Some(vec![Rc::clone(&self.0)]),
        )
    }
}

impl Add for &Tensor {
    type Output = Tensor;

    fn add(self, other: Self) -> Self::Output {
        let data = &self.0.borrow().data + &other.0.borrow().data;

        Tensor::new(
            data,
            Operation::Add,
            Some(vec![Rc::clone(&self.0), Rc::clone(&other.0)]),
        )
    }
}

impl Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        let data = -&self.0.borrow().data;
        Tensor::new(data, Operation::Neg, Some(vec![Rc::clone(&self.0)]))
    }
}

impl Sub for &Tensor {
    type Output = Tensor;

    fn sub(self, other: Self) -> Self::Output {
        let data = &self.0.borrow().data - &other.0.borrow().data;

        Tensor::new(
            data,
            Operation::Sub,
            Some(vec![Rc::clone(&self.0), Rc::clone(&other.0)]),
        )
    }
}

impl Mul for &Tensor {
    type Output = Tensor;

    fn mul(self, other: Self) -> Self::Output {
        let data = self.0.borrow().data.elemul(&other.0.borrow().data);

        Tensor::new(
            data,
            Operation::Mul,
            Some(vec![Rc::clone(&self.0), Rc::clone(&other.0)]),
        )
    }
}

trait Sum {
    type Output;
    fn sum(self, dim: usize) -> Self::Output;
}

impl Sum for &Tensor {
    type Output = Tensor;

    fn sum(self, axis: usize) -> Self::Output {
        if axis > 1 {
            unimplemented!();
        }

        let ans = if axis == 0 {
            let data = &self.0.borrow().data;
            let mut summed_data = Matrix::zeros(1, data.cols());
            for i in 0..data.cols() {
                for j in 0..data.rows() {
                    summed_data[[0, i]] += data[[j, i]];
                }
            }
            summed_data
        } else {
            let data = &self.0.borrow().data;
            let mut summed_data = Matrix::zeros(data.rows(), 1);
            for i in 0..data.rows() {
                for j in 0..data.cols() {
                    summed_data[[i, 0]] += data[[i, j]];
                }
            }
            summed_data
        };

        Tensor::new(ans, Operation::Sum(axis), Some(vec![Rc::clone(&self.0)]))
    }
}

trait Expand {
    type Output;
    fn expand(self, dim: usize, copies: usize) -> Self::Output;
}

impl Expand for &Tensor {
    type Output = Tensor;

    fn expand(self, dim: usize, copies: usize) -> Self::Output {
        if dim == 0 {
            let new_data = {
                let data = &self.0.borrow().data;
                if data.rows() != 1 {
                    unimplemented!()
                }

                let mut new_data = Matrix::zeros(copies, data.cols());
                for i in 0..copies {
                    for j in 0..data.cols() {
                        new_data[[i, j]] = data[[0, j]];
                    }
                }

                new_data
            };

            Tensor::new(
                new_data,
                Operation::Expand(dim),
                Some(vec![Rc::clone(&self.0)]),
            )
        } else {
            unimplemented!()
        }
    }
}

trait Transpose {
    type Output;
    fn transpose(self) -> Self::Output;
}

impl Transpose for &Tensor {
    type Output = Tensor;

    fn transpose(self) -> Self::Output {
        let res = {
            let data = &self.0.borrow().data;
            data.transpose()
        };
        Tensor::new(res, Operation::Transpose, Some(vec![Rc::clone(&self.0)]))
    }
}

trait Dot {
    type Output;
    fn dot(self, other: Self) -> Self::Output;
}

impl Dot for &Tensor {
    type Output = Tensor;

    fn dot(self, other: &Tensor) -> Self::Output {
        let result = {
            let data = &self.0.borrow().data;
            let other_data = &other.0.borrow().data;
            data.mul(other_data)
        };

        Tensor::new(
            result,
            Operation::Dot,
            Some(vec![Rc::clone(&self.0), Rc::clone(&other.0)]),
        )
    }
}

/// Using Autograd to train a Neural Network

fn training_using_autograd() {
    let data = Tensor::new_const(Matrix::new(
        4,
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    ));
    let target = Tensor::new_const(Matrix::new(4, 1, vec![0.0, 1.0, 0.0, 1.0]));

    let distribution = Uniform::new(0.0, 1.0);

    let w1 = Tensor::new_const(Matrix::new(
        2,
        3,
        generate_random_vector(2 * 3, 1.0, 0.0, &distribution),
    ));
    let w2 = Tensor::new_const(Matrix::new(
        3,
        1,
        generate_random_vector(3, 1.0, 0.0, &distribution),
    ));

    let alpha = 0.1;

    for _ in 0..10 {
        let pred = data.dot(&w1).dot(&w2);
        let loss = (&(&pred - &target) * &(&pred - &target)).sum(0);
        let (loss_rows, loss_cols) = (1, 1);

        println!("Loss: {:?}", loss.0.borrow().data);

        loss.backward(Tensor::grad(Matrix::ones(loss_rows, loss_cols)));

        {
            let mut w1 = w1.0.borrow_mut();
            let grad = w1.grad.take();
            w1.grad = None;

            let grad = grad.unwrap();
            let grad = &grad.borrow().data;

            for i in 0..w1.data.rows() {
                for j in 0..w1.data.cols() {
                    w1.data[[i, j]] -= alpha * grad[[i, j]];
                }
            }
        }

        {
            let mut w2 = w2.0.borrow_mut();
            let grad = w2.grad.take();
            w2.grad = None;

            let grad = grad.unwrap();
            let grad = &grad.borrow().data;

            for i in 0..w2.data.rows() {
                for j in 0..w2.data.cols() {
                    w2.data[[i, j]] -= alpha * grad[[i, j]];
                }
            }
        }
    }
}

/// Adding Automatic Optimization

fn training_with_automatic_optimization() {
    let data = Tensor::new_const(Matrix::new(
        4,
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    ));
    let target = Tensor::new_const(Matrix::new(4, 1, vec![0.0, 1.0, 0.0, 1.0]));

    let distribution = Uniform::new(0.0, 1.0);

    let w1 = Tensor::new_const(Matrix::new(
        2,
        3,
        generate_random_vector(2 * 3, 1.0, 0.0, &distribution),
    ));

    let w2 = Tensor::new_const(Matrix::new(
        3,
        1,
        generate_random_vector(3, 1.0, 0.0, &distribution),
    ));

    let alpha = 0.1;

    let optimizer = SGDOptimizer::new(vec![&w1, &w2], alpha);

    for _ in 0..10 {
        // predict
        let pred = data.dot(&w1).dot(&w2);

        // compare
        let loss = (&(&pred - &target) * &(&pred - &target)).sum(0);
        let (loss_rows, loss_cols) = (1, 1);

        println!("Loss: {:?}", loss.0.borrow().data.data());

        // calculate difference
        loss.backward(Tensor::grad(Matrix::ones(loss_rows, loss_cols)));

        // learn
        optimizer.step(true);
    }
}

trait Optimizer {
    fn step(&self, zero: bool);
}

struct SGDOptimizer<'a> {
    parameters: Vec<&'a Tensor>,
    alpha: f64,
}

impl<'a> SGDOptimizer<'a> {
    fn new(parameters: Vec<&'a Tensor>, alpha: f64) -> SGDOptimizer {
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

trait Layer {
    fn forward(&self, inputs: &[&Tensor]) -> Vec<Tensor>;

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }
}

struct Linear {
    weights: Tensor,
    bias: Tensor,
}

impl Linear {
    fn new(n_inputs: usize, n_outputs: usize) -> Linear {
        let distribution = Uniform::new(0.0, 1.0);

        let weights = Tensor::new_const(Matrix::new(
            n_inputs,
            n_outputs,
            generate_random_vector(n_inputs * n_outputs, 0.5, 0.0, &distribution),
        ));

        let bias = Tensor::new_const(Matrix::zeros(1, n_outputs));

        Linear { weights, bias }
    }
}

impl Layer for Linear {
    fn forward(&self, inputs: &[&Tensor]) -> Vec<Tensor> {
        let rows = inputs[0].0.borrow().data.rows();
        vec![&inputs[0].dot(&self.weights) + &self.bias.expand(0, rows)]
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weights, &self.bias]
    }
}

/// Layers Which Contain Layers

fn layers_which_contain_layers() {
    let data = Tensor::new_const(Matrix::new(
        4,
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    ));

    let target = Tensor::new_const(Matrix::new(4, 1, vec![0.0, 1.0, 0.0, 1.0]));

    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 3)),
        Box::new(Linear::new(3, 1)),
    ]);

    let optim = SGDOptimizer::new(model.parameters(), 0.05);

    for _ in 0..10 {
        let pred = model.forward(&data);

        // compare
        let loss = (&(&pred[0] - &target) * &(&pred[0] - &target)).sum(0);

        println!("Loss: {:?}", loss.0.borrow().data.data());

        // calculate difference
        loss.backward(Tensor::grad(Matrix::ones(1, 1)));

        // learn
        optim.step(true);
    }
}

struct Sequential {
    layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        Sequential { layers }
    }

    #[allow(dead_code)]
    fn add(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    fn forward(&self, input: &Tensor) -> Vec<Tensor> {
        // TODO: can this be avoided
        let mut input = Tensor(Rc::clone(&input.0));

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

fn loss_function_layers() {
    let data = Tensor::new_const(Matrix::new(
        4,
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    ));

    let target = Tensor::new_const(Matrix::new(4, 1, vec![0.0, 1.0, 0.0, 1.0]));

    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 3)),
        Box::new(Linear::new(3, 1)),
    ]);

    let criterion = MSELoss;
    let optim = SGDOptimizer::new(model.parameters(), 0.05);

    for _ in 0..10 {
        let pred = model.forward(&data);

        // compare
        let loss = criterion.forward(&pred[0], &target);

        println!("Loss: {:?}", loss.0.borrow().data.data());

        // calculate difference
        loss.backward(Tensor::grad(Matrix::ones(1, 1)));

        // learn
        optim.step(true);
    }
}

trait Loss {
    fn forward(&self, pred: &Tensor, target: &Tensor) -> Tensor;
}

struct MSELoss;

impl Loss for MSELoss {
    fn forward(&self, pred: &Tensor, target: &Tensor) -> Tensor {
        (&(pred - target) * &(pred - target)).sum(0)
    }
}

struct Sigmoid;

impl Layer for Sigmoid {
    fn forward(&self, inputs: &[&Tensor]) -> Vec<Tensor> {
        vec![inputs[0].sigmoid()]
    }
}

struct Tanh;

impl Layer for Tanh {
    fn forward(&self, inputs: &[&Tensor]) -> Vec<Tensor> {
        vec![inputs[0].tanh()]
    }
}

#[allow(dead_code)]
struct Relu;

impl Layer for Relu {
    fn forward(&self, inputs: &[&Tensor]) -> Vec<Tensor> {
        vec![inputs[0].relu()]
    }
}

/// NonLinearity Layers

fn nonlinearity_layers() {
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
        let pred = model.forward(&data);

        // compare
        let loss = criterion.forward(&pred[0], &target);

        println!("Loss: {:?}", loss.0.borrow().data.data());

        // calculate difference
        loss.backward(Tensor::grad(Matrix::ones(1, 1)));

        // learn
        optim.step(true);
    }
}

struct Embedding {
    weights: Tensor,
}

impl Embedding {
    fn new(vocab_size: usize, embedding_size: usize) -> Embedding {
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

/// The Embedding Layer

fn embedding_layer() {
    let data = Tensor::new_const(Matrix::new(1, 4, vec![1.0, 2.0, 1.0, 2.0]));
    let target = Tensor::new_const(Matrix::new(4, 1, vec![0.0, 1.0, 0.0, 1.0]));

    let model = Sequential::new(vec![
        Box::new(Embedding::new(5, 3)),
        Box::new(Tanh),
        Box::new(Linear::new(3, 1)),
        Box::new(Sigmoid),
    ]);

    let criterion = MSELoss;
    let optim = SGDOptimizer::new(model.parameters(), 0.07);

    for _ in 0..10 {
        let pred = model.forward(&data);

        // compare
        let loss = criterion.forward(&pred[0], &target);

        println!("Loss: {:?}", loss.0.borrow().data.data());

        // calculate difference
        loss.backward(Tensor::grad(Matrix::ones(1, 1)));

        // learn
        optim.step(true);
    }
}

/// The Cross Entropy Layer

fn cross_entropy_loss() {
    let data = Tensor::new_const(Matrix::new(1, 4, vec![1.0, 2.0, 1.0, 2.0]));
    let target = Tensor::new_const(Matrix::new(4, 1, vec![0.0, 1.0, 0.0, 1.0]));

    let model = Sequential::new(vec![
        Box::new(Embedding::new(3, 3)),
        Box::new(Tanh),
        Box::new(Linear::new(3, 4)),
    ]);

    let criterion = CrossEntropyLoss;
    let optim = SGDOptimizer::new(model.parameters(), 0.1);

    for _ in 0..10 {
        let pred = model.forward(&data);
        // println!("pred {}", pred.0.borrow().data);

        // compare
        let loss = criterion.forward(&pred[0], &target);

        println!("Loss: {:?}", loss.0.borrow().data.data());

        // calculate difference
        loss.backward(Tensor::grad(Matrix::ones(1, 1)));

        // learn
        optim.step(true);
    }
}

struct CrossEntropyLoss;

impl Loss for CrossEntropyLoss {
    fn forward(&self, pred: &Tensor, target_indices: &Tensor) -> Tensor {
        pred.cross_entropy(target_indices)
    }
}

#[allow(clippy::needless_range_loop)]
fn recurrent_neural_network() -> Result<(), Box<dyn Error>> {
    let (train_data, _) = babi_en_single_supporting_fact_task()?;

    let train_data: Vec<Vec<String>> = train_data
        .map(|v| vec![v.0, v.1 /*, (v.2).0*/])
        .flat_map(|v| v.into_iter())
        .map(|s| {
            s.split_whitespace()
                .map(|w| {
                    w.chars()
                        .filter(|c| (*c >= 'a' && *c <= 'z') || (*c >= 'A' && *c <= 'Z'))
                        .collect()
                })
                .collect()
        })
        .collect();

    let total_data_size = train_data.len();

    let words = BTreeSet::from_iter(train_data.iter().flat_map(|v| v.iter()));

    let word_count = words.len();
    let word_index = BTreeMap::from_iter(words.into_iter().zip(0..word_count));
    let inverted_word_index =
        BTreeMap::from_iter(word_index.clone().into_iter().map(|(k, v)| (v, k)));

    let train_data: Vec<Vec<f64>> = train_data
        .iter()
        .map(|s| s.iter().map(|w| word_index[w] as f64).collect())
        .collect();

    let max_len = train_data.iter().map(|s| s.len()).max().unwrap();
    let pad = word_index.len() + 1;

    let batch_size = 250;

    let train_data: Vec<_> = train_data
        .into_iter()
        .batch(batch_size, true)
        .map(|v: Vec<Vec<f64>>| {
            let mut ans = vec![vec![0.0; batch_size]; max_len];
            for i in 0..batch_size {
                for j in 0..v[i].len() {
                    ans[j][i] = v[i][j];
                }

                for j in v[i].len()..max_len {
                    ans[j][i] = pad as f64;
                }
            }

            ans
        })
        .collect();

    let embedding_size = 16;

    // net
    let embed = Embedding::new(word_index.len() + 2, embedding_size);
    let model = RNNCell::new(embedding_size, 16, word_index.len() + 2, Box::new(Sigmoid));

    let criterion = CrossEntropyLoss;
    let mut parameters = embed.parameters();
    parameters.append(&mut model.parameters());

    let optim = SGDOptimizer::new(parameters, 0.01);

    for _ in 0..10 {
        let mut total_loss = 0.0;
        let mut total_accuracy = 0.0;

        for batch in train_data.iter() {
            let mut hidden = model.create_start_state(batch_size);
            let mut output = None;

            let len = batch.len();

            for row in batch.iter().take(len - 1) {
                let input = Tensor::new_const(Matrix::new(1, batch_size, row.clone()));
                let rnn_input = embed.forward(&[&input]).remove(0);
                let mut outputs = model.forward(&[&rnn_input, &hidden]);
                output = Some(outputs.remove(0));
                hidden = outputs.remove(0);
            }

            let output = output.unwrap();

            let target = Tensor::new_const(Matrix::new(batch_size, 1, batch[len - 1].clone()));

            let loss = criterion.forward(&output, &target);
            loss.backward(Tensor::new_const(Matrix::ones(1, 1)));

            optim.step(true);

            let current_loss = loss.0.borrow().data.data()[0];
            total_loss += current_loss;

            let current_accuracy: f64 = output
                .0
                .borrow()
                .data
                .row_iter()
                .zip(batch[len - 1].iter())
                .map(|(row, ix)| {
                    if argmax(row.raw_slice()) == (*ix) as usize {
                        1.0
                    } else {
                        0.0
                    }
                })
                .sum();

            total_accuracy += current_accuracy;
        }

        println!(
            "Loss: {}, Accuracy: {}",
            total_loss,
            total_accuracy / (total_data_size as f64)
        );
    }

    let batch = vec![
        vec![word_index[&"Mary".to_owned()] as f64],
        vec![word_index[&"moved".to_owned()] as f64],
        vec![word_index[&"to".to_owned()] as f64],
        vec![word_index[&"the".to_owned()] as f64],
    ];

    let mut hidden = model.create_start_state(1);
    let mut output = None;
    for row in batch.iter() {
        let input = Tensor::new_const(Matrix::new(1, 1, row.clone()));
        let rnn_input = embed.forward(&[&input]).remove(0);
        let mut outputs = model.forward(&[&rnn_input, &hidden]);
        output = Some(outputs.remove(0));
        hidden = outputs.remove(0);
    }

    let output = argmax(output.unwrap().0.borrow().data.row(0).raw_slice());
    println!("Prediction: {}", inverted_word_index[&output]);

    Ok(())
}

#[allow(dead_code)]
struct RNNCell {
    n_inputs: usize,
    n_hidden: usize,
    n_outputs: usize,
    w_ih: Linear,
    w_hh: Linear,
    w_ho: Linear,
    activation: Box<dyn Layer>,
}

impl RNNCell {
    fn new(
        n_inputs: usize,
        n_hidden: usize,
        n_outputs: usize,
        activation: Box<dyn Layer>,
    ) -> RNNCell {
        let w_ih = Linear::new(n_inputs, n_hidden);
        let w_hh = Linear::new(n_hidden, n_hidden);
        let w_ho = Linear::new(n_hidden, n_outputs);

        RNNCell {
            n_inputs,
            n_hidden,
            n_outputs,
            w_ih,
            w_hh,
            w_ho,
            activation,
        }
    }

    fn create_start_state(&self, batch_size: usize) -> Tensor {
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
