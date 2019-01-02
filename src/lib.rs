pub type Vector = Vec<f64>;
pub type Matrix = Vec<Vec<f64>>;

pub fn elementwise_multiplication(vec_a: &Vector, vec_b: &Vector) -> Vector {
    vec_a
        .iter()
        .zip(vec_b.iter())
        .map(|(a, b)| a * b)
        .collect()
}

pub fn vector_sum(vec: Vector) -> f64 {
    vec.iter().sum()
}

pub fn dot(vec_a: &Vector, vec_b: &Vector) -> f64 {
    elementwise_multiplication(vec_a, vec_b).iter().sum()
}

pub fn elementwise_scalar_multiplication(vec: &Vector, n: f64) -> Vector {
    vec.iter().map(|x| x * n).collect()
}

pub fn elementwise_addition(vec_a: &Vector, vec_b: &Vector) -> Vector {
    vec_a
        .iter()
        .zip(vec_b.iter())
        .map(|(a, b)| a + b)
        .collect()
}

pub fn vector_average(vec: &Vector) -> f64 {
    let len = vec.len() as f64;
    vec.iter().sum::<f64>() / len
}

pub fn matrix_vector_multiplication(mat: &Matrix, vec: &Vector) -> Vector {
    mat.iter().map(|w| dot(w, vec)).collect()
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
    fn test_vector_matrix_multiplication() {
        assert_eq!(
            vec![55.0, 45.0, 40.0, 40.0, 35.0],
            matrix_vector_multiplication(
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
}
