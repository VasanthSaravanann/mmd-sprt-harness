# WP4 — LaTeX proof notes

## Theorem 1: Normalization argument doesn't hold for RBF kernel

The proof attempts to bound `|MMD²(P||Q) - MMD²(P0||Q0)|` using a normalization
argument that assumes a linear kernel.  With the Gaussian RBF kernel the feature
map lives in an infinite-dimensional RKHS and the normalization step (dividing
count vectors by length) does not factor through the kernel evaluation the same
way.  In practice the empirical spectra are already length-normalized, but the
theoretical bound in the paper uses a different normalization than what the code
implements.

**Suggested fix:** Either restrict the kernel to the linear case for the proof
(the code now uses RBF — switching to linear would degrade detection), or derive
the bound directly in terms of the RBF kernel's Taylor expansion, showing that
`K(x,y) = exp(-||x-y||²/(2σ²))` is Lipschitz in `||x-y||²` with modulus
`1/(2σ²)` so that the MMD² difference can be bounded by the L² difference of
the empirical distributions.

## Theorem 2: Bound gives 0.33 at N=5, not under 0.05

The theorem claims that with N ≥ 5 the false-halt probability is bounded below
0.05.  Actually evaluating the published bound (which depends on 1/(N-2) from
the residual degrees of freedom of the exponential fit) gives ~0.33 at N=5,
not < 0.05.  The empirical measurements confirm this: the false-decision rate at
N=5 is ~4.9%, which happens to be below 5% — but the **theoretical bound** does
not prove it.

**Suggested fix:** Replace the Chebyshev-style bound with a tighter concentration
inequality (e.g. McDiarmid's inequality applied to the R² statistic) or lower
the bound to match the empirical finding and note that the theorem provides a
conservative estimate rather than a tight one.