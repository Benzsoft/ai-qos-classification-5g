# Research foundation — start here

This addition preserves the collaborator's six families: Random Forest, SVM, KNN, LSTM, BiLSTM, and IP embeddings. Original source, license, models, and attribution remain unchanged. The original BiLSTM builder has no attention layer despite the old README description.

## Colab sequence

1. Open [00 setup](https://colab.research.google.com/github/Benzsoft/ai-qos-classification-5g/blob/research/colab-foundation/notebooks/00_colab_setup.ipynb). Select T4 GPU and run cells in order. Approve Drive mounting when prompted.
2. Open [01 audit](https://colab.research.google.com/github/Benzsoft/ai-qos-classification-5g/blob/research/colab-foundation/notebooks/01_dataset_audit.ipynb) and run cells in order. It works in its own runtime and remounts Drive as needed.
3. Share the printed audit JSON. Outputs persist under MyDrive/5G_QoS_Research; local Colab checkout is temporary. Do not post credentials, raw IP addresses, or private captures in GitHub issues.

No dependency upgrades are performed. Your observed Colab environment was Python 3.13.15, TensorFlow 2.20.0, scikit-learn 1.6.1, pandas 2.2.3, NumPy 2.1.3. Notebook 00 captures the actual environment and checkout SHA on every run; this is not a cross-platform lockfile. Existing checkouts are reused without pulling or resetting changes. A fresh checkout uses this branch's current tip; recorded SHAs must be frozen for final experiments.

## Dataset candidates and outstanding checks

- [Korean 5G traffic](https://www.kaggle.com/datasets/kimdaegyeom/5g-traffic-datasets): primary real-5G application candidate. Pending acquisition, file inventory, license, labels, and session audit. No real data has been downloaded by these notebooks.
- [CESNET DataZoo](https://cesnet.github.io/cesnet-datazoo/): temporal and unknown-service evaluation. Backbone data must not be described as measured 5G. Official documentation says default S has 25 million samples, so no automatic full download. Select a bounded acquisition approach after inspecting metadata. Service/domain labels must not be silently converted to activity labels.
- [ISCXVPN2016](https://www.unb.ca/cic/datasets/vpn.html): older application pilot; verify access, license, capture boundaries, and label mapping. Full-flow aggregate CSVs may contain information unavailable at early prediction time; prefer packet histories when available.

For every selected source record provider, version, URL/DOI, license, file sizes, checksums, timestamp units, label provenance, identifier availability, and independent session counts. The CSV audit is bounded to a prefix and is not an acquisition adapter for PCAP or HDF5.

## Two tasks, two papers

Task A: observed application/service labels, evaluated separately per dataset. Cross-dataset tests require compatible labels and features.
Task B: explicitly configured simulated eMBB/URLLC/mMTC service profiles. Do not infer service requirements from packet size or map voice automatically to URLLC. Report these as simulation ground truth.

Paper 1: reproducible six-model benchmark, chronological shift, equal observation budgets, feature ablations, and compute cost. A ranking alone does not establish novelty.
Paper 2: new risk-aware act/wait/fallback policy and closed-loop QoS experiments. Compare the same classifiers under the same controller and network conditions. Cite/disclose shared work and distinguish principal experiments. Neither manuscript nor results exist yet.

## Experimental gates

1. Audit provenance and independently labeled sessions.
2. Partition sessions/time/scenarios into train/validation/test BEFORE windows; embargo overlapping time boundaries where required.
3. Fit encoders/scalers only on training; handle unknown IDs explicitly. Keep the IP-embedding family, but mark unavailable-ID datasets not applicable rather than inventing IDs.
4. Give all six equal observation budgets and shared target instances. Use ordered packets for recurrent models and flattened/summarized prefixes for classical models. No future/full-flow features. A BiLSTM can only see the already-observed prefix.
5. Tune on validation with declared budgets. Start with seeds 11, 23, 37, 51, 71; use shared sampled sessions for the primary comparison. Pilot sizes depend on memory/time measurements.
6. Report macro-F1, balanced accuracy, per-class recall, calibration, train time, batch-one inference, memory and observation latency. Bootstrap independent sessions/runs, not correlated packets. Keep timing comparisons on matched hardware.
7. Simulate mixed broadband/control/sensing traffic using a pinned and verified ns-3/5G-LENA release. Check traffic source vs observed-network timing to avoid double-counting transport effects. Include observation, inference, and actuation delay.
8. Compare static/conventional control, immediate classification, fixed observation, confidence waiting, proposed risk policy, and oracle labels. Measure deadline misses, tail latency, goodput, fairness and overhead with paired scenario seeds.

## Eight-week working schedule

Weeks 1–2: acquisition audit and six-model pilot. Weeks 3–4: tuning and main benchmark. Weeks 5–6: robustness, Paper 1 analysis, basic simulator. Weeks 7–8: Paper 1 submission checks and Paper 2 policy experiments. Two submissions are conditional on sufficient evidence; no acceptance or completion guarantee.

## Current validation boundary

The CSV audit is locally runnable. Colab-specific Drive mounting and T4 execution must be verified in the user's runtime. These notebooks do not yet implement the corrected six-model training, real-data adapters, risk controller, or network simulator. They establish the setup and audit milestone without presenting synthetic diagnostics as scientific results.
