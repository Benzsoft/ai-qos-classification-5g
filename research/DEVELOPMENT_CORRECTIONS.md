# Development correction protocol

Notebook 08 preserves the original pilot. All existing held-out captures are now development data because their results informed subsequent choices. No untouched final test is currently available. Do not describe this experiment as final confirmation of distribution-shift robustness.

## Verified extraction diagnosis
YouTube training captures yield 6, 5, and 5 eligible segments from 174, 148, and 573 inactivity segments. No invalid feature rows or reservoir losses occurred. The 16 retained segments share one recording-date group. Training captures have zero leading port hints; development captures have approximately 79% and 90%. This is evidence of export/traffic representation differences, not proof of a causal explanation. Changing inactivity thresholds or making ongoing windows solely to increase sample count would change the experimental unit.

## Prespecified sensitivity study
Seeds: 42, 43, 44. Prefix budgets: 5, 10, 20, 30. Six original model families retained. Four conditions: raw protocols/unbalanced; numeric/unbalanced; numeric/balanced; numeric/balanced/zero endpoint inputs. Balancing retains every original training sample and samples minority classes with replacement to the majority count. It is the same training-only procedure for all models, including KNN. Scalers and vocabularies fit the original training partition before resampling. No holdout resampling or tuning occurs. IP identity ablation preserves the model architecture. It does not make the IP model identical to the other architectures.

Tables report accuracy, macro-F1, balanced accuracy, and YouTube recall. Error bars show standard deviation across seeds, not confidence intervals over independent recordings. No statistical significance or hardware-neutral speed ranking is claimed. Earlier per-model latency estimates exclude observation/preprocessing costs and mix CPU/GPU execution.

All budgets use the same cohort of segments that reach 30 packets. This isolates prefix information but cannot measure coverage of short conversations. Date groups are independence proxies, not verified sessions/devices. Predictions and per-model metrics remain available per run; keep endpoint maps and raw features private.

## Still required for a publishable evaluation
1. Obtain documented reuse terms and recording/session/device provenance from the dataset source.
2. Acquire additional independently recorded applications/sessions, especially YouTube; keep related captures together. Existing records cannot be made independent by resampling.
3. Reserve untouched groups before model/parameter selection. If available groups cannot support train/validation/final test, expand the dataset instead of splitting packets randomly.
4. Review packet exports for transport identifiers and consistent protocol dissection before claiming five-tuple flow classification.
5. Evaluate uncertainty over independent groups once there are enough groups per class. Three model seeds alone are inadequate.
6. Implement and validate the network simulator separately, including oracle, static/default, and classifier-based QoS policies, classification observation delay, and confidence/fallback decisions. No QoS improvement has been measured yet.
7. Write conclusions only from completed results; do not infer publication readiness from improved development scores.

## Verification
Training-only scaling, unknown-category handling, prefix slicing, and nonmutation were checked using local fixtures. Notebook code cells passed syntax parsing. TensorFlow execution of the new study must be verified in Colab; no revised performance results are asserted here.
