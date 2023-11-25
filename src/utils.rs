use burn::tensor::backend::Backend;
use burn::tensor::{Data, Element, ElementConversion, Numeric, Shape, Tensor};

// https://github.com/huggingface/transformers/blob/674f750a57431222fa2832503a108df3badf1564/src/transformers/models/clip/modeling_clip.py#L678
pub(crate) fn generate_causal_attention_mask<B: Backend>(
    bsz: usize,
    seq_len: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let mut mask_vec = Vec::with_capacity(seq_len * seq_len);
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                mask_vec.push(f32::MIN.elem());
            } else {
                mask_vec.push(0f32.elem());
            }
        }
    }

    let mask_data: Data<B::FloatElem, 2> = Data::new(mask_vec, Shape::new([seq_len, seq_len]));
    Tensor::from_data_device(mask_data, device)
        .unsqueeze::<3>()
        .repeat(0, bsz)
        .unsqueeze_dim(1)
}

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

// TODO: Remove once https://github.com/Tracel-AI/burn/pull/998 is merged
pub(crate) fn chunk<B, const D: usize, K>(
    tensor: Tensor<B, D, K>,
    chunks: usize,
    dim: usize,
) -> Vec<Tensor<B, D, K>>
where
    B: Backend,
    K: Numeric<B>,
    K::Elem: Element,
{
    if dim >= D {
        panic!("dim must be less than the number of dimensions of the tensor");
    }

    let size = tensor.shape().dims[dim];
    if size < chunks {
        (0..size)
            .map(|i| tensor.clone().narrow(dim, i, 1))
            .collect::<Vec<_>>()
    } else {
        let chunk_size = size / chunks;
        let cnt_additional = size % chunks;
        let mut tensors = vec![];
        let mut sum_chunk_size = 0;
        for i in 0..chunks {
            let chunk_size = if i < cnt_additional {
                chunk_size + 1
            } else {
                chunk_size
            };
            let tensor = tensor.clone().narrow(dim, sum_chunk_size, chunk_size);
            tensors.push(tensor);
            sum_chunk_size += chunk_size
        }
        tensors
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::backend::Backend;
    use burn::tensor::{Data, Int, Shape};

    #[test]
    fn test_build_causal_attention_mask() {
        type TestBackend = burn_ndarray::NdArray<f32>;
        let device = <TestBackend as Backend>::Device::default();

        let mask = generate_causal_attention_mask::<TestBackend>(2, 4, &device);
        assert_eq!(mask.shape(), Shape::from([2, 1, 4, 4]));

        mask.to_data().assert_approx_eq(
            &Data::from([
                [[
                    [0.0000e0, f32::MIN, f32::MIN, f32::MIN],
                    [0.0000e0, 0.0000e0, f32::MIN, f32::MIN],
                    [0.0000e0, 0.0000e0, 0.0000e0, f32::MIN],
                    [0.0000e0, 0.0000e0, 0.0000e0, 0.0000e0],
                ]],
                [[
                    [0.0000e0, f32::MIN, f32::MIN, f32::MIN],
                    [0.0000e0, 0.0000e0, f32::MIN, f32::MIN],
                    [0.0000e0, 0.0000e0, 0.0000e0, f32::MIN],
                    [0.0000e0, 0.0000e0, 0.0000e0, 0.0000e0],
                ]],
            ]),
            3,
        );
    }

    #[test]
    fn test_pad_with_zeros() {
        type TestBackend = burn_ndarray::NdArray<f32>;

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

    #[test]
    fn test_chunk_odd_divisible() {
        type TestBackend = burn_ndarray::NdArray<f32>;

        let tensor: Tensor<TestBackend, 1, Int> = Tensor::arange(0..11);
        let tensors = chunk(tensor, 6, 0);

        assert_eq!(tensors.len(), 6);

        let expected = vec![
            Data::from([0, 1]),
            Data::from([2, 3]),
            Data::from([4, 5]),
            Data::from([6, 7]),
            Data::from([8, 9]),
            Data::from([10]),
        ];

        for (index, tensor) in tensors.iter().enumerate() {
            assert_eq!(tensor.to_data(), expected[index]);
        }
    }

    #[test]
    fn test_chunk_even_divisible() {
        type TestBackend = burn_ndarray::NdArray<f32>;

        let tensor: Tensor<TestBackend, 1, Int> = Tensor::arange(0..12);
        let tensors = chunk(tensor, 6, 0);

        assert_eq!(tensors.len(), 6);

        let expected = vec![
            Data::from([0, 1]),
            Data::from([2, 3]),
            Data::from([4, 5]),
            Data::from([6, 7]),
            Data::from([8, 9]),
            Data::from([10, 11]),
        ];

        for (index, tensor) in tensors.iter().enumerate() {
            assert_eq!(tensor.to_data(), expected[index]);
        }
    }

    #[test]
    fn test_chunk_not_divisible() {
        type TestBackend = burn_ndarray::NdArray<f32>;

        let tensor: Tensor<TestBackend, 1, Int> = Tensor::arange(0..4);
        let tensors = chunk(tensor, 6, 0);

        assert_eq!(tensors.len(), 4);

        let expected = vec![
            Data::from([0]),
            Data::from([1]),
            Data::from([2]),
            Data::from([3]),
        ];

        for (index, tensor) in tensors.iter().enumerate() {
            assert_eq!(tensor.to_data(), expected[index]);
        }
    }

    #[test]
    fn test_chunk_multi_dimension() {
        type TestBackend = burn_ndarray::NdArray<f32>;

        let tensor: Tensor<TestBackend, 2, Int> = Tensor::from_data(Data::from([[0, 1, 2, 3]]));
        let tensors = chunk(tensor, 2, 1);

        assert_eq!(tensors.len(), 2);

        let expected = vec![Data::from([[0, 1]]), Data::from([[2, 3]])];

        for (index, tensor) in tensors.iter().enumerate() {
            assert_eq!(tensor.to_data(), expected[index]);
        }
    }
}
