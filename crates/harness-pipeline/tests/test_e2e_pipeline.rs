use harness_core::gate0::Stream;
use harness_pipeline::{run_pipeline, dummy_fuzzer, DeployVerdict, SystemSpec};
use rand::{rngs::SmallRng, Rng, SeedableRng};

const MOTIFS: &[&[u8]] = &[b"abba", b"aaba", b"baab"];
const SECRET: u8 = b'X';

fn motif_stream(motif: &[u8], len: usize) -> Stream {
    (0..len).map(|i| motif[i % motif.len()]).collect()
}

fn make_prod() -> Vec<Stream> {
    let mut rng = SmallRng::seed_from_u64(123);
    (0..90)
        .map(|_| {
            let motif = MOTIFS[rng.gen_range(0..MOTIFS.len())];
            motif_stream(motif, 60)
        })
        .collect()
}

fn healthy(prod: &[Stream]) -> Vec<Stream> {
    let mut rng = SmallRng::seed_from_u64(124);
    prod.iter()
        .map(|s| {
            let mut s = s.clone();
            for _ in 0..(s.len() as f64 * 0.002).round() as usize {
                let i = rng.gen_range(0..s.len());
                s[i] = if rng.gen_bool(0.5) { b'a' } else { b'b' };
            }
            s
        })
        .collect()
}

fn exfil(prod: &[Stream]) -> Vec<Stream> {
    let mut rng = SmallRng::seed_from_u64(125);
    prod.iter()
        .map(|s| {
            let mut s = s.clone();
            for _ in 0..(s.len() as f64 * 0.01).round().max(1.0) as usize {
                let i = rng.gen_range(0..s.len());
                s[i] = SECRET;
            }
            s
        })
        .collect()
}

fn gen(seed: u64, n: usize, r: f64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let xs: Vec<f64> = (0..n).map(|_| rng.gen_range(0.0..1.0)).collect();
    let ys: Vec<f64> = xs.iter().map(|x| r * x + (1.0 - r) * rng.gen_range(0.0..0.1)).collect();
    (xs, ys)
}

fn good_ipw() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let (px, py) = gen(1, 3000, 0.90);
    let (ex, ey) = gen(2, 4000, 0.95);
    let plen = px.len();
    let elen = ex.len();
    (px, py, vec![0.01; plen], ex, ey, vec![0.4; elen])
}

fn bad_ipw() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let (px, py, pp, ex, _ey, ep) = good_ipw();
    let noise: Vec<f64> = (0..ex.len()).map(|_| 0.5).collect();
    (px, py, pp, ex, noise, ep)
}

fn spec(prod: &[Stream], cand: &[Stream], sprt: &[f64],
        ipw: (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)) -> SystemSpec {
    let (px, py, pp, ex, ey, ep) = ipw;
    SystemSpec {
        prod_streams: prod.to_vec(),
        cand_streams: cand.to_vec(),
        sprt_distances: sprt.to_vec(),
        ipw_prod_x: px, ipw_prod_y: py, ipw_prod_sel: pp,
        ipw_eval_x: ex, ipw_eval_y: ey, ipw_eval_sel: ep,
    }
}

#[test]
fn robust_system_grants_full_deployment() {
    let prod = make_prod();
    let s = spec(&prod, &healthy(&prod), &[0.30, 0.20, 0.12, 0.07, 0.04], good_ipw());
    let rep = run_pipeline(&s, dummy_fuzzer().as_ref());
    assert_eq!(rep.verdict, DeployVerdict::FullDeploymentGranted, "{}", rep.stage_summary());
    assert!(rep.gate0.is_some());
    assert!(rep.sprt.is_some());
    assert!(rep.evaluator.is_some());
}

#[test]
fn flawed_baseline_halts_deployment() {
    let prod = make_prod();
    let s = spec(&prod, &exfil(&prod), &[0.30, 0.10, 0.28, 0.05, 0.22], bad_ipw());
    let rep = run_pipeline(&s, dummy_fuzzer().as_ref());
    assert_eq!(rep.verdict, DeployVerdict::DeploymentHalted);
    assert_eq!(rep.verdict.as_str(), "DEPLOYMENT_HALTED");
    assert!(!rep.gate0.as_ref().unwrap().passed);
}

#[test]
fn single_stage_weakness_halts_whole_pipeline() {
    let prod = make_prod();
    // All good except SPRT is jagged.
    let s = spec(&prod, &healthy(&prod), &[0.30, 0.10, 0.28, 0.05, 0.22], good_ipw());
    let rep = run_pipeline(&s, dummy_fuzzer().as_ref());
    assert_eq!(rep.verdict, DeployVerdict::DeploymentHalted);
    assert!(rep.gate0.as_ref().unwrap().passed);
    assert!(!rep.sprt.as_ref().unwrap().converged);
}

#[test]
fn pipeline_verdict_strings() {
    assert_eq!(DeployVerdict::FullDeploymentGranted.as_str(), "FULL_DEPLOYMENT_GRANTED");
    assert_eq!(DeployVerdict::DeploymentHalted.as_str(), "DEPLOYMENT_HALTED");
}