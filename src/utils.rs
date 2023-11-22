use burn::tensor::backend::Backend;
use burn::tensor::{Data, ElementConversion, Shape, Tensor};

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
}
