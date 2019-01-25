//! This was extracted from the Chapter 13 exercises and moved into the core library so it could be used in later chapters.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ops::{Add, Mul, Neg, Sub};
use std::rc::Rc;

use rand::{thread_rng, RngCore};
use rulinalg::matrix::{BaseMatrix, Matrix};

pub type TensorRef = Rc<RefCell<TensorImpl>>;

#[derive(Debug, Clone)]
pub enum Operation {
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

#[derive(Debug)]
pub struct TensorImpl {
    id: u64,
    pub data: Matrix<f64>,
    pub grad: Option<TensorRef>,
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
                None => Some(Rc::clone(&grad)),
                Some(current_grad) => {
                    let new_grad_data = {
                        let current_grad_data = &current_grad.borrow().data;
                        let grad_data = &grad.borrow().data;
                        current_grad_data + grad_data
                    };

                    Some(Rc::new(RefCell::new(TensorImpl::grad(new_grad_data))))
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
                            creators[1].borrow_mut().backward(
                                Rc::new(RefCell::new(TensorImpl::grad(-data))),
                                Some(self.id),
                            );
                        }
                    }
                    Operation::Mul => {
                        let grad = &grad.borrow().data;

                        let grad0 = {
                            let grad0 = &creators[1].borrow().data;
                            let grad0 = grad0.elemul(grad);
                            Rc::new(RefCell::new(TensorImpl::grad(grad0)))
                        };

                        let grad1 = {
                            let grad1 = &creators[0].borrow().data;
                            let grad1 = grad1.elemul(grad);
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
                            let data = &self.data;
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
                            let data = &self.data;
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
                            let data = &self.data;
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
                                    new_grad[[*ix, j]] += grad[[i, j]];
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
pub struct Tensor(pub TensorRef);

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Tensor(Rc::clone(&self.0))
    }
}

impl Tensor {
    pub fn new_const(data: Matrix<f64>) -> Self {
        Self::new(data, Operation::Const, None)
    }

    pub fn grad(data: Matrix<f64>) -> Self {
        let tensor_impl = TensorImpl::grad(data);
        Tensor(Rc::new(RefCell::new(tensor_impl)))
    }

    pub fn new(
        data: Matrix<f64>,
        creation_op: Operation,
        creators: Option<Vec<TensorRef>>,
    ) -> Self {
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

    pub fn backward(&self, grad: Tensor) {
        self.0.borrow_mut().backward(grad.0, None);
    }

    /// higher order ops

    pub fn sigmoid(&self) -> Tensor {
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

        if self.0.borrow().autograd {
            Tensor::new(result, Operation::Sigmoid, Some(vec![Rc::clone(&self.0)]))
        } else {
            Tensor::grad(result)
        }
    }

    pub fn tanh(&self) -> Tensor {
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

        if self.0.borrow().autograd {
            Tensor::new(result, Operation::Tanh, Some(vec![Rc::clone(&self.0)]))
        } else {
            Tensor::grad(result)
        }
    }

    pub fn relu(&self) -> Tensor {
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

        if self.0.borrow().autograd {
            Tensor::new(result, Operation::Relu, Some(vec![Rc::clone(&self.0)]))
        } else {
            Tensor::grad(result)
        }
    }

    pub fn index_select(&self, indices: Vec<usize>) -> Tensor {
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

        if self.0.borrow().autograd {
            Tensor::new(
                result,
                Operation::IndexSelect(indices),
                Some(vec![Rc::clone(&self.0)]),
            )
        } else {
            Tensor::grad(result)
        }
    }

    /// the current tensor and the targets have to be the same shape
    pub fn cross_entropy(&self, target_indices: &Tensor) -> Tensor {
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

            loss /= data.rows() as f64;

            (m, target_dist, loss)
        };

        if self.0.borrow().autograd {
            Tensor::new(
                Matrix::new(1, 1, vec![loss]),
                Operation::CrossEntropy(m, target_dist),
                Some(vec![Rc::clone(&self.0)]),
            )
        } else {
            Tensor::grad(Matrix::new(1, 1, vec![loss]))
        }
    }
}

impl Add for &Tensor {
    type Output = Tensor;

    fn add(self, other: Self) -> Self::Output {
        let data = &self.0.borrow().data + &other.0.borrow().data;

        if self.0.borrow().autograd {
            Tensor::new(
                data,
                Operation::Add,
                Some(vec![Rc::clone(&self.0), Rc::clone(&other.0)]),
            )
        } else {
            Tensor::grad(data)
        }
    }
}

impl Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        let data = -&self.0.borrow().data;
        if self.0.borrow().autograd {
            Tensor::new(data, Operation::Neg, Some(vec![Rc::clone(&self.0)]))
        } else {
            Tensor::grad(data)
        }
    }
}

impl Sub for &Tensor {
    type Output = Tensor;

    fn sub(self, other: Self) -> Self::Output {
        let data = &self.0.borrow().data - &other.0.borrow().data;

        if self.0.borrow().autograd {
            Tensor::new(
                data,
                Operation::Sub,
                Some(vec![Rc::clone(&self.0), Rc::clone(&other.0)]),
            )
        } else {
            Tensor::grad(data)
        }
    }
}

impl Mul for &Tensor {
    type Output = Tensor;

    fn mul(self, other: Self) -> Self::Output {
        let data = self.0.borrow().data.elemul(&other.0.borrow().data);

        if self.0.borrow().autograd {
            Tensor::new(
                data,
                Operation::Mul,
                Some(vec![Rc::clone(&self.0), Rc::clone(&other.0)]),
            )
        } else {
            Tensor::grad(data)
        }
    }
}

pub trait Sum {
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

        if self.0.borrow().autograd {
            Tensor::new(ans, Operation::Sum(axis), Some(vec![Rc::clone(&self.0)]))
        } else {
            Tensor::grad(ans)
        }
    }
}

pub trait Expand {
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

            if self.0.borrow().autograd {
                Tensor::new(
                    new_data,
                    Operation::Expand(dim),
                    Some(vec![Rc::clone(&self.0)]),
                )
            } else {
                Tensor::grad(new_data)
            }
        } else {
            unimplemented!()
        }
    }
}

pub trait Transpose {
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

        if self.0.borrow().autograd {
            Tensor::new(res, Operation::Transpose, Some(vec![Rc::clone(&self.0)]))
        } else {
            Tensor::grad(res)
        }
    }
}

pub trait Dot {
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

        if self.0.borrow().autograd {
            Tensor::new(
                result,
                Operation::Dot,
                Some(vec![Rc::clone(&self.0), Rc::clone(&other.0)]),
            )
        } else {
            Tensor::grad(result)
        }
    }
}
