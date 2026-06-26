# Bitter lessons — ROCm KDA-pilot kernel optimization loops

Hard-won, reusable lessons from running the Humanize/KDA optimization loops on FlyDSL kernels
(gfx950 FlashAttention so far). Each entry is a specific failure mode + the specific fix, so the
next loop on a different kernel can avoid re-paying the tuition. Sources are the per-loop reports
in `results/`.

---

## Methodology (apply to every loop)

### BL-01 — A single-point benchmark misleads systematically
**Where it bit:** loop-07 (fp8 wide-MFMA). Six rounds optimized and reported against ONE config
(`B1 S2048 H64 nocausal`), concluding "~82% of the reference." A representative multi-shape sweep,
run only after external pushback, flipped the story: **causal beats the reference (geomean 117%,
up to 151%); non-causal trails (geomean 68%, down to 56% at large seqlen).** The single point was
a favorable sample hiding two opposite conclusions.
**Fix:** make a representative multi-shape sweep (full config set × {causal, non-causal}, varied
B/S/H/GQA) a **gate** for any perf claim, not a final nicety. Report geomean + win-count + range
per regime; never a lone headline number. FLOPs must be causal-discounted so causal vs non-causal
TFLOPS are comparable.

### BL-02 — Persist every cited number as an immutable artifact
**Where it bit:** loop-07 round 1. Headline TFLOPS were cited from a shared, run-overwritten CSV;
a later run clobbered it and the persisted file no longer matched the claim. The numbers were
real; the provenance was gone, and review rejected it.
**Fix:** the moment a number is produced, write it to an immutable, per-mode, per-run-named file
and cite that exact path. Never rely on a tool's default shared output filename for a cited number.

### BL-03 — Close dead-end levers with falsification, don't hand-wave "future work"
**Where it bit:** loop-04/05 (incremental tuning) and loop-07 (wide-PV). Risk is either tuning a
dead end forever, or deferring with no evidence (which strict review treats as incomplete).
**Fix:** before deferring a lever, either attempt it OR produce explicit falsification evidence
(a cheap negative benchmark, or a profiled binding-constraint argument). "Closed with
falsification evidence" is a legitimate terminal state. Examples: a prior loop used 7 probes to
prove barrier/vmcnt/occupancy tuning was exhausted; loop-07 measured the wide-PV combos as
slower-than-default (tested-negative) instead of saying "maybe later."

### BL-04 — Before blaming your change, run a control group
**Where it bit:** loop-07 local CI verification. The PR-branch test suite showed 31 failures + 4
collection errors — looked like our change broke things. Running the **same suite + same build on
clean upstream/main** proved 100% of them were a stale shared build + unrelated upstream merges,
not our change.
**Fix:** for any "did my change break X?" question, reproduce on a clean baseline (same
build/runtime) first, diff the failing-test sets, and cross-check `git diff <base>..HEAD
--name-only` to confirm your diff even touches the relevant files.

---

## Technical (FlyDSL / gfx950 / fp8)

### BL-05 — Bottleneck ≠ instruction width: profile first
**Where it bit:** loop-06/07. Adopting a 4× wider MMA cut MFMA instruction count 4× but moved
wall-clock only ~5–9% on the default path. The kernel was not MFMA-issue-bound: roofline showed
HBM ~8–10% utilized, ~41% of fp8 MFMA peak; the real constraint was a VGPR-limited dual-wave
pipeline (occupancy/latency-bound).
**Fix:** profile for the actual binding constraint (roofline + GRBM active cycles + occupancy/VGPR
alloc + stall bubbles) before investing. A wider/faster instruction helps only when *that
instruction's* issue throughput is the bottleneck; otherwise the lever is structural (occupancy
or schedule rewrite). Attention also can't hit dense-GEMM peak (softmax/exp/rescale overhead).

### BL-06 — Prove equivalence with ISA byte-identity, not "looks equivalent"
**Where it bit:** loop-07 (the gate-off non-regression claim and the helper refactor). Two near
misses: a permlane32 lane-half bug (a correct-looking layout that fed half the lanes wrong values,
min_cos 0.939); and a "pure refactor" that changed final ISA because two independent pure ops were
reordered.
**Fix:** validate refactors/gating with **final-ISA byte-identity** (dump the lowered ISA on HEAD
vs base, compare md5) plus the on-device correctness gate — not by inspection. When reordering
independent pure ops for readability, keep the original eval order (named locals) so the emitted
ISA stays byte-identical.

### BL-07 — Decode operand layouts on-device, not on paper
**Where it bit:** loop-07. The wide `32x32x64` MMA's in-register operand layout differs from 4×
narrow `32x32x16`; paper derivations produced wrong intermediate states (min_cos 0.804/0.886).
**Fix:** decode the operand layout with an on-device "operand-oracle" probe (drive the MMA with
known inputs, read back the (lane,byte)→(row/col, contraction-index) map), and treat the
end-to-end correctness gate as the ground truth.

### BL-08 — One optimization = one default-off gate (= bisectable + safe + attributable)
**Where it bit (positively):** loop-07. Every lever was a separate env gate, so "toggle gate =
commit bisection" gave exact attribution (the entire default-path gain was the wide QK atom; wide
PV only helped the slow NATIVE mode) without a real git bisect, and default-off kept delivery
byte-identical.
**Fix:** keep each optimization behind its own default-off gate; it makes attribution, safe
default delivery, and ISA-identity verification trivial.

### BL-09 — Converge exploratory mode/flag sprawl
**Where it bit:** loop-04→07. The fp8 path accumulated three PV precision modes (HIPREC default +
FROMBF16/NATIVE opt-in) plus several wide-atom gates. NATIVE/FROMBF16 are neither fastest nor
default — exploration-era comparison paths that became a cognitive tax ("why are there so many
modes?").
**Fix:** treat mode/flag sprawl as a readability signal; converge or clearly document
opt-in/experimental paths before handoff, so the next reader isn't taxed.

### BL-10 — Deliver to the PR branch's real head
**Where it bit:** loop-07 delivery. The local worktree tracking the PR branch was stale vs the real
PR head; a naive merge would have pulled unrelated upstream commits.
**Fix:** query the real head (`gh pr view <n> --json headRefOid`), re-anchor (`git reset --hard
<head>`) before applying, verify only intended files stage, validate at the delivered head, then
push and reference the actual commit SHA + an evidence gist in the PR comment.
