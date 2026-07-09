/// JOLT Atlas example: prove the injection-probe ONNX inference.
///
/// Placement: jolt-atlas-core/examples/probe.rs
/// Model:     atlas-onnx-tracer/models/probe/network.onnx
///
/// Run:
///   cargo run --release --package jolt-atlas-core --example probe -- --trace-terminal
///
/// Patterned on examples/bge.rs (closest structural match: a feature-vector
/// model with explicit prove timing). The probe takes one 1536-length vector,
/// shape [1, 1536].
use atlas_onnx_tracer::{
    model::{Model, RunArgs},
    tensor::Tensor,
};
use common::utils::logging::setup_tracing;
use jolt_atlas_core::onnx_proof::{
    AtlasProverPreprocessing, AtlasSharedPreprocessing, AtlasVerifierPreprocessing,
    Blake2bTranscript, Bn254, Fr, HyperKZG, ONNXProof,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn main() {
    let (_guard, _tracing_enabled) = setup_tracing("probe");

    let dim: usize = 1536;
    let run_args = RunArgs::new([("batch_size", 1)]);
    let model = Model::load("atlas-onnx-tracer/models/probe/network.onnx", &run_args);
    println!("{}", model.pretty_print());
    println!("max num vars: {}", model.max_num_vars());

    // Probe input: one activation vector as [m=1, k=dim] (rank-2). The exported
    // graph uses an explicit einsum `mk,nk->n` with a padded weight [n=2, k], so
    // the input must carry the m dimension. i32 values (JOLT Atlas quantizes
    // internally). Placeholder values; swap in a real quantized activation row.
    let mut rng = StdRng::seed_from_u64(0x1096);
    let input_data: Vec<i32> = (0..dim).map(|_| rng.gen_range(-128..=128)).collect();
    let input = Tensor::new(Some(&input_data), &[1, dim]).unwrap();

    tracing::info!("Loaded probe model and generated input");
    let pp = AtlasSharedPreprocessing::preprocess(model);
    let prover_preprocessing = AtlasProverPreprocessing::<Fr, HyperKZG<Bn254>>::new(pp);

    let timing = std::time::Instant::now();
    let (proof, io, _debug_info) = ONNXProof::<Fr, Blake2bTranscript, HyperKZG<Bn254>>::prove(
        &prover_preprocessing,
        &[input],
    );
    println!("Proof generation took {:.2?}", timing.elapsed());

    let verifier_preprocessing =
        AtlasVerifierPreprocessing::<Fr, HyperKZG<Bn254>>::from(&prover_preprocessing);

    let vtiming = std::time::Instant::now();
    proof.verify(&verifier_preprocessing, &io, None).unwrap();
    println!("Proof verification took {:.2?}", vtiming.elapsed());

    println!("Proof verified successfully!");
}
