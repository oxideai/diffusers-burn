use burn::tensor::backend::Backend;
use burn::tensor::{Data, Element, ElementConversion, Numeric, Shape, Tensor};

// https://github.com/huggingface/transformers/blob/674f750a57431222fa2832503a108df3badf1564/src/transformers/models/clip/modeling_clip.py#L678
pub(crate) fn build_causal_attention_mask<B: Backend>(
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::backend::Backend;
    use burn::tensor::{Data, Shape};

    #[test]
    fn test_build_causal_attention_mask() {
        type TestBackend = burn_ndarray::NdArray<f32>;
        let device = <TestBackend as Backend>::Device::default();

        let mask = build_causal_attention_mask::<TestBackend>(2, 4, &device);
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
        let device = <TestBackend as Backend>::Device::default();

        let tensor: Tensor<TestBackend, 3> = Tensor::from_data_device(
            Data::from([[[1.6585, 0.4320], [-0.8701, -0.4649]]]),
            &device,
        );

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
