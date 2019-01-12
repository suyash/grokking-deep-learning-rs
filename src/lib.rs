pub type Vector = Vec<f64>;
pub type Matrix = Vec<Vec<f64>>;

pub fn elementwise_multiplication(vec_a: &Vector, vec_b: &Vector) -> Vector {
    vec_a.iter().zip(vec_b.iter()).map(|(a, b)| a * b).collect()
}

pub fn argmax(vec: &[f64]) -> usize {
    let mut max = vec[0];
    let mut ans = 0;

    for i in 1..vec.len() {
        if vec[i] > max {
            max = vec[i];
            ans = i;
        }
    }

    return ans;
}

pub fn vector_sum(vec: Vector) -> f64 {
    vec.iter().sum()
}

pub fn dot(vec_a: &Vector, vec_b: &Vector) -> f64 {
    vec_a.iter().zip(vec_b.iter()).map(|(a, b)| a * b).sum()
}

pub fn elementwise_scalar_multiplication(vec: &Vector, n: f64) -> Vector {
    vec.iter().map(|x| x * n).collect()
}

pub fn elementwise_addition(vec_a: &Vector, vec_b: &Vector) -> Vector {
    vec_a.iter().zip(vec_b.iter()).map(|(a, b)| a + b).collect()
}

pub fn vector_average(vec: &Vector) -> f64 {
    let len = vec.len() as f64;
    vec.iter().sum::<f64>() / len
}

pub fn vector_vector_subtraction(v1: &Vector, v2: &Vector) -> Vector {
    v1.iter().zip(v2.iter()).map(|(a, b)| a - b).collect()
}

pub fn vector_vector_multiplication(v1: &Vector, v2: &Vector) -> Vector {
    v1.iter().zip(v2.iter()).map(|(a, b)| a * b).collect()
}

pub fn vector_vector_dot(vec1: &Vector, vec2: &Vector) -> Matrix {
    vec1.iter()
        .map(|i| vec2.iter().map(|j| i * j).collect())
        .collect()
}

pub fn vector_matrix_dot(vec: &Vector, mat: &Matrix) -> Vector {
    matrix_vector_dot(&transpose(mat), vec)
}

pub fn matrix_vector_dot(mat: &Matrix, vec: &Vector) -> Vector {
    mat.iter().map(|w| dot(w, vec)).collect()
}

pub fn matrix_matrix_subtraction(mat1: &Matrix, mat2: &Matrix) -> Matrix {
    mat1.iter()
        .zip(mat2.iter())
        .map(|(v1, v2)| vector_vector_subtraction(v1, v2))
        .collect()
}

pub fn matrix_matrix_multiplication(mat1: &Matrix, mat2: &Matrix) -> Matrix {
    mat1.iter()
        .zip(mat2.iter())
        .map(|(v1, v2)| vector_vector_multiplication(v1, v2))
        .collect()
}

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
    m.into_iter().map(|v| relu_vector(v)).collect()
}

pub fn relu_matrix_derivative(m: Matrix) -> Matrix {
    m.into_iter().map(|v| relu_vector_derivative(v)).collect()
}

pub fn transpose(m: &Matrix) -> Matrix {
    let mut ans = vec![vec![0.0; m.len()]; m[0].len()];

    for i in 0..m.len() {
        for j in 0..m[0].len() {
            ans[j][i] = m[i][j];
        }
    }

    ans
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
