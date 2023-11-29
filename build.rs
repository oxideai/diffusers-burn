fn main() {
    // Check if any of the features are specified
    let any_feature_selected = cfg!(any(feature = "ndarray", feature = "ndarray-no-std", feature = "torch", feature = "wgpu"));
    let has_std = cfg!(feature = "std");

    // If none of the features are specified, default to wgpu
    if !any_feature_selected {
        if !has_std {
            println!("cargo:rerun-if-env-changed=FORCE_NDARRAY_NO_STD"); // Optional: Trigger a recompile if needed
            println!("cargo:rustc-cfg=ndarray-no-std");
        } else {
            println!("cargo:rerun-if-env-changed=FORCE_WGPU"); // Optional: Trigger a recompile if needed
            println!("cargo:rustc-cfg=wgpu");
        }
    }
}
