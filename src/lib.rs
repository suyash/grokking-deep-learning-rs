use datasets::Dataset;
use rand::distributions::{Bernoulli, Distribution};
use rand::{thread_rng, Rng};
use rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix as RulinalgMatrix};

pub type Vector = Vec<f64>;
pub type Matrix = Vec<Vec<f64>>;

#[allow(clippy::ptr_arg)]
pub fn elementwise_multiplication(vec_a: &Vector, vec_b: &Vector) -> Vector {
    vec_a.iter().zip(vec_b.iter()).map(|(a, b)| a * b).collect()
}

pub fn argmax(vec: &[f64]) -> usize {
    let mut max = vec[0];
    let mut ans = 0;

    for (i, x) in vec.iter().enumerate().skip(1) {
        if x > &max {
            max = *x;
            ans = i;
        }
    }

    ans
}

pub fn vector_sum(vec: Vector) -> f64 {
    vec.iter().sum()
}

#[allow(clippy::ptr_arg)]
pub fn dot(vec_a: &Vector, vec_b: &Vector) -> f64 {
    vec_a.iter().zip(vec_b.iter()).map(|(a, b)| a * b).sum()
}

#[allow(clippy::ptr_arg)]
pub fn elementwise_scalar_multiplication(vec: &Vector, n: f64) -> Vector {
    vec.iter().map(|x| x * n).collect()
}

#[allow(clippy::ptr_arg)]
pub fn elementwise_addition(vec_a: &Vector, vec_b: &Vector) -> Vector {
    vec_a.iter().zip(vec_b.iter()).map(|(a, b)| a + b).collect()
}

#[allow(clippy::ptr_arg)]
pub fn vector_average(vec: &Vector) -> f64 {
    let len = vec.len() as f64;
    vec.iter().sum::<f64>() / len
}

#[allow(clippy::ptr_arg)]
pub fn vector_vector_subtraction(v1: &Vector, v2: &Vector) -> Vector {
    v1.iter().zip(v2.iter()).map(|(a, b)| a - b).collect()
}

#[allow(clippy::ptr_arg)]
pub fn vector_vector_multiplication(v1: &Vector, v2: &Vector) -> Vector {
    v1.iter().zip(v2.iter()).map(|(a, b)| a * b).collect()
}

#[allow(clippy::ptr_arg)]
pub fn vector_vector_dot(vec1: &Vector, vec2: &Vector) -> Matrix {
    vec1.iter()
        .map(|i| vec2.iter().map(|j| i * j).collect())
        .collect()
}

#[allow(clippy::ptr_arg)]
pub fn vector_matrix_dot(vec: &Vector, mat: &Matrix) -> Vector {
    matrix_vector_dot(&transpose(mat), vec)
}

#[allow(clippy::ptr_arg)]
pub fn matrix_vector_dot(mat: &Matrix, vec: &Vector) -> Vector {
    mat.iter().map(|w| dot(w, vec)).collect()
}

#[allow(clippy::ptr_arg)]
pub fn matrix_matrix_subtraction(mat1: &Matrix, mat2: &Matrix) -> Matrix {
    mat1.iter()
        .zip(mat2.iter())
        .map(|(v1, v2)| vector_vector_subtraction(v1, v2))
        .collect()
}

#[allow(clippy::ptr_arg)]
pub fn matrix_matrix_multiplication(mat1: &Matrix, mat2: &Matrix) -> Matrix {
    mat1.iter()
        .zip(mat2.iter())
        .map(|(v1, v2)| vector_vector_multiplication(v1, v2))
        .collect()
}

#[allow(clippy::ptr_arg, clippy::needless_range_loop)]
pub fn matrix_matrix_dot(mat1: &Matrix, mat2: &Matrix) -> Matrix {
    assert_eq!(mat1[0].len(), mat2.len());

    let mut ans = vec![vec![0.0; mat2[0].len()]; mat1.len()];

    for i in 0..mat1.len() {
        for j in 0..mat2[0].len() {
            for k in 0..mat2.len() {
                ans[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }

    ans
}

pub fn relu_vector(v: Vector) -> Vector {
    v.into_iter()
        .map(|a| if a > 0.0 { a } else { 0.0 })
        .collect()
}

pub fn relu_vector_derivative(v: Vector) -> Vector {
    v.into_iter()
        .map(|a| if a > 0.0 { 1.0 } else { 0.0 })
        .collect()
}

pub fn relu_matrix(m: Matrix) -> Matrix {
    m.into_iter().map(relu_vector).collect()
}

pub fn relu_matrix_derivative(m: Matrix) -> Matrix {
    m.into_iter().map(relu_vector_derivative).collect()
}

#[allow(clippy::ptr_arg, clippy::needless_range_loop)]
pub fn transpose(m: &Matrix) -> Matrix {
    let mut ans = vec![vec![0.0; m.len()]; m[0].len()];

    for i in 0..m.len() {
        for j in 0..m[0].len() {
            ans[j][i] = m[i][j];
        }
    }

    ans
}

pub fn generate_random_vector(
    size: usize,
    scale_factor: f64,
    add_factor: f64,
    dist: &impl Distribution<f64>,
) -> Vec<f64> {
    let mut rng = thread_rng();
    (0..size)
        .map(|_| scale_factor * rng.sample(dist) + add_factor)
        .collect()
}

pub fn process_mnist_batch_dataset(
    dataset: impl Dataset<Item = (Vec<u8>, u8)>,
    dataset_size: usize,
    batch_size: usize,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let normalize_image = |img: Vec<u8>| img.iter().map(|v| f64::from(*v) / 255.0).collect();
    let encode_label = |l| {
        let mut v = vec![0.0; 10];
        v[l as usize] = 1.0;
        v
    };

    let (images, labels): (Vec<_>, Vec<_>) = dataset
        .take(dataset_size)
        .map(|(i, l)| (normalize_image(i), encode_label(l)))
        .unzip();

    let images = images
        .into_iter()
        .batch(batch_size, false)
        .map(|v| {
            v.into_iter()
                .fold(Vec::with_capacity(batch_size * 784), |mut acc, mut img| {
                    acc.append(&mut img);
                    acc
                })
        })
        .collect();

    let labels = labels
        .into_iter()
        .batch(batch_size, false)
        .map(|v| {
            v.into_iter()
                .fold(Vec::with_capacity(batch_size * 10), |mut acc, mut l| {
                    acc.append(&mut l);
                    acc
                })
        })
        .collect();

    (images, labels)
}

pub fn sample_bernoulli_trials(p: f64, length: usize) -> Vec<f64> {
    let dist = Bernoulli::new(p);
    thread_rng()
        .sample_iter(&dist)
        .take(length)
        .map(|v| if v { 1.0 } else { 0.0 })
        .collect()
}

pub fn relu_mut(m: &mut RulinalgMatrix<f64>) {
    for x in m.iter_mut() {
        *x = if (*x) > 0.0 { *x } else { 0.0 };
    }
}

pub fn relu_derivative(m: &RulinalgMatrix<f64>) -> RulinalgMatrix<f64> {
    let mut ans = RulinalgMatrix::zeros(m.rows(), m.cols());
    for i in 0..m.rows() {
        for j in 0..m.cols() {
            if m[[i, j]] >= 0.0 {
                ans[[i, j]] = 1.0;
            }
        }
    }

    ans
}

pub fn sigmoid_mut(m: &mut RulinalgMatrix<f64>) {
    for x in m.iter_mut() {
        *x = 1.0 / (1.0 + (-(*x)).exp());
    }
}

pub fn tanh_mut(m: &mut RulinalgMatrix<f64>) {
    for x in m.iter_mut() {
        *x = (*x).tanh();
    }
}

pub fn tanh_derivative(m: &RulinalgMatrix<f64>) -> RulinalgMatrix<f64> {
    let mut ans = RulinalgMatrix::zeros(m.rows(), m.cols());
    for i in 0..m.rows() {
        for j in 0..m.cols() {
            ans[[i, j]] = 1.0 - (m[[i, j]] * m[[i, j]]);
        }
    }
    ans
}

pub fn softmax_mut(m: &mut RulinalgMatrix<f64>) {
    for i in 0..m.rows() {
        let mut s = 0.0;

        for j in 0..m.cols() {
            m[[i, j]] = m[[i, j]].exp();
            s += m[[i, j]];
        }

        for j in 0..m.cols() {
            m[[i, j]] /= s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elementwise_multiplication() {
        assert_eq!(
            vec![6.0, 14.0, 24.0, 36.0, 0.0],
            elementwise_multiplication(
                &vec![1.0, 2.0, 3.0, 4.0, 5.0],
                &vec![6.0, 7.0, 8.0, 9.0, 0.0],
            ),
        );
    }

    #[test]
    fn test_vector_sum() {
        assert_eq!(15.0, vector_sum(vec![1.0, 2.0, 3.0, 4.0, 5.0]));
    }

    #[test]
    fn test_elementwise_addition() {
        assert_eq!(
            vec![7.0, 9.0, 11.0, 13.0, 5.0],
            elementwise_addition(
                &vec![1.0, 2.0, 3.0, 4.0, 5.0],
                &vec![6.0, 7.0, 8.0, 9.0, 0.0],
            ),
        )
    }

    #[test]
    fn test_vector_average() {
        assert_eq!(3.0, vector_average(&vec![1.0, 2.0, 3.0, 4.0, 5.0]));
    }

    #[test]
    fn test_dot() {
        assert_eq!(
            80.0,
            dot(
                &vec![1.0, 2.0, 3.0, 4.0, 5.0],
                &vec![6.0, 7.0, 8.0, 9.0, 0.0],
            ),
        );
    }

    #[test]
    fn test_elementwise_scalar_multiplication() {
        assert_eq!(
            vec![2.0, 4.0, 6.0, 8.0, 10.0],
            elementwise_scalar_multiplication(&vec![1.0, 2.0, 3.0, 4.0, 5.0], 2.0,)
        )
    }

    #[test]
    fn test_matrix_vector_dot() {
        assert_eq!(
            vec![55.0, 45.0, 40.0, 40.0, 35.0],
            matrix_vector_dot(
                &vec![
                    vec![1.0, 2.0, 3.0, 4.0, 5.0],
                    vec![2.0, 3.0, 4.0, 5.0, 1.0],
                    vec![3.0, 4.0, 5.0, 1.0, 2.0],
                    vec![4.0, 5.0, 1.0, 2.0, 3.0],
                    vec![5.0, 4.0, 3.0, 2.0, 1.0],
                ],
                &vec![1.0, 2.0, 3.0, 4.0, 5.0],
            ),
        );
    }

    #[test]
    fn test_relu_vector() {
        assert_eq!(
            vec![1.0, 0.0, 2.0, 0.0, 4.0],
            relu_vector(vec![1.0, -1.0, 2.0, -2.0, 4.0]),
        );
    }
}
