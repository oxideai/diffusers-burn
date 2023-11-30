use alloc::vec;
use burn::tensor::backend::Backend;
use burn::tensor::{Element, Numeric, Tensor};

pub(crate) fn pad_with_zeros<B, const D: usize, K>(
    tensor: Tensor<B, D, K>,
    dim: usize,
    left: usize,
    right: usize,
) -> Tensor<B, D, K>
where
    B: Backend,
    K: Numeric<B>,
    K::Elem: Element,
{
    if left == 0 && right == 0 {
        tensor.clone()
    } else if left == 0 {
        assert!(
            dim < D,
            "dim must be less than the number of dimensions of the tensor"
        );
        let mut dims = tensor.shape().dims.to_vec();
        dims[dim] = right;
        let right = Tensor::zeros_device(dims, &tensor.device());
        Tensor::cat(vec![tensor, right], dim)
    } else if right == 0 {
        assert!(
            dim < D,
            "dim must be less than the number of dimensions of the tensor"
        );
        let mut dims = tensor.shape().dims.to_vec();
        dims[dim] = left;
        let left = Tensor::zeros_device(dims, &tensor.device());
        Tensor::cat(vec![left, tensor], dim)
    } else {
        assert!(
            dim < D,
            "dim must be less than the number of dimensions of the tensor"
        );
        let mut dims = tensor.shape().dims.to_vec();
        dims[dim] = left;
        let left = Tensor::zeros_device(dims.clone(), &tensor.device());
        dims[dim] = right;
        let right = Tensor::zeros_device(dims, &tensor.device());
        Tensor::cat(vec![left, tensor, right], dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn::tensor::{Data, Shape};

    #[test]
    fn test_pad_with_zeros() {
        let tensor: Tensor<TestBackend, 3> =
            Tensor::from_data(Data::from([[[1.6585, 0.4320], [-0.8701, -0.4649]]]));

        let padded = pad_with_zeros(tensor, 0, 1, 2);

        assert_eq!(padded.shape(), Shape::from([4, 2, 2]));
        padded.to_data().assert_approx_eq(
            &Data::from([
                [[0.0000, 0.0000], [0.0000, 0.0000]],
                [[1.6585, 0.4320], [-0.8701, -0.4649]],
                [[0.0000, 0.0000], [0.0000, 0.0000]],
                [[0.0000, 0.0000], [0.0000, 0.0000]],
            ]),
            3,
        )
    }
}
