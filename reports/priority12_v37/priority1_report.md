# Báo cáo Priority 1 — V37 Q1a XLM-R Large

**Trạng thái tổng thể: PASS**
**Phạm vi:** component ablation của recipe Q1a V37 trên XLM-R Large
**Audit snapshot:** 03/10/2026 UTC (04/10/2026 theo giờ Việt Nam)

## 1. Tóm tắt điều hành

Priority 1 đã hoàn tất đúng recipe V37 và cho kết quả hợp lệ:

- 4 ablation variants × 3 seeds = **12/12 run hợp lệ**.
- **12/12 local manifests và protocol validations PASS**.
- **12/12 HF artifact receipts PASS**, mỗi run có 22 file artifact.
- Aggregate report: **PASS**, gồm 4 paired-bootstrap comparisons.
- Không còn training runner, aggregate job, P2 hoặc P3 process đang chạy.
- Các kiểm tra code và test contract đều PASS.

Full V37 trong báo cáo là **frozen reference đã được audit trước**, không chạy lại trong campaign ablation này. Các run mới chỉ bỏ từng thành phần để đo ảnh hưởng tương đối so với full.

## 2. Mục tiêu và các biến thể

Mục tiêu là kiểm tra đóng góp của bốn thành phần trong recipe V37:

| Variant | Thành phần bị bỏ |
|---|---|
| `full` | Reference V37 đầy đủ, dùng làm đối chứng |
| `no_emotion_auxiliary` | Emotion auxiliary loss |
| `no_polarity_auxiliary` | Polarity auxiliary loss |
| `no_rationale` | Rationale supervision/decoder |
| `no_uncertainty_weighting` | Uncertainty weighting |

Mỗi ablation dùng cùng dataset split, cùng batch order, cùng training protocol và ba seed `20260521`, `20260522`, `20260523`. Không có chọn seed theo test score.

## 3. Setup thực nghiệm đã dùng

### Model và dữ liệu

- Model: `FacebookAI/xlm-roberta-large`
- Model/tokenizer revision cố định: `c23d21b0620b635a76227c604d44e43a9f0ee389`
- Maximum sequence length: 128
- Split: train 7,998; dev 1,999; test 2,000
- Thiết bị được ghi nhận: NVIDIA H100 80GB HBM3, MIG 3g.40gb

### Training recipe

- Optimizer: AdamW, learning rate `2e-5`, weight decay `0.01`
- Scheduler: cosine, warmup ratio `0.20`
- Tối đa 10 epochs, patience 10, minimum delta `0.0001`
- Physical batch 8, effective batch 32, gradient accumulation 4
- BF16, gradient clipping 1.0, batch order cố định
- Rationale beta `0.3`
- Threshold chọn trên dev; test chỉ chạy sau khi checkpoint và threshold đã freeze

### V37 head multipliers

| Head | Multiplier |
|---|---:|
| implicit sentiment | 1.01 |
| sarcasm | 1.01 |
| irony | 1.05 |
| idiom/figurative | 1.00 |
| code-switching | 1.04 |
| mocking | 1.00 |
| polarity/emotion | 1.00 |

Đây là **V37 Q1a full XLM-R Large recipe**, không phải profile V22 cũ.

## 4. Kết quả chính

Các giá trị là trung bình trên 3 seed; SD được biểu diễn theo percentage point (pp) đối với F1.

| Variant | Macro pragmatic F1 | Sarcasm binary macro F1 | Pragmatic ECE |
|---|---:|---:|---:|
| **Full V37 reference** | **93.6658% ± 0.1871 pp** | **89.3606% ± 0.8970 pp** | 0.038591 ± 0.006938 |
| No emotion auxiliary | 92.9310% ± 0.5019 pp | 87.8686% ± 0.5840 pp | **0.028590 ± 0.001589** |
| No polarity auxiliary | 92.7494% ± 0.2366 pp | 87.6035% ± 1.3169 pp | 0.032925 ± 0.003083 |
| No rationale | 92.9864% ± 0.2520 pp | 87.0194% ± 1.0131 pp | 0.033200 ± 0.001699 |
| No uncertainty weighting | 93.2642% ± 0.0815 pp | 88.3463% ± 1.1701 pp | 0.038689 ± 0.006613 |

### Độ lệch so với Full: paired bootstrap

So sánh dùng `Full − Ablation`, 1,000 resamples, bootstrap seed `20260525`; CI là khoảng tin cậy bootstrap trên macro pragmatic F1.

| So sánh | Chênh lệch trung bình | 95% bootstrap CI | Diễn giải |
|---|---:|---:|---|
| Full − No emotion | +0.7348 pp | [-0.0614, +1.6246] pp | Point estimate có lợi cho full, CI còn chạm 0 |
| Full − No polarity | **+0.9164 pp** | **[+0.4803, +1.4463] pp** | Bằng chứng rõ nhất trong bốn ablation |
| Full − No rationale | +0.6795 pp | [+0.2086, +1.1629] pp | CI dương |
| Full − No uncertainty | +0.4016 pp | [-0.0811, +0.8699] pp | Gần full nhất; CI còn chạm 0 |

## 5. Diễn giải

1. **Full V37 vẫn là cấu hình tốt nhất** trên metric chính: 93.6658% macro pragmatic F1.
2. **Polarity auxiliary** có tín hiệu đóng góp lớn nhất: bỏ thành phần này làm giảm trung bình 0.9164 pp và CI không chứa 0 trong paired bootstrap.
3. **Rationale supervision** cũng có tác động dương rõ trong thí nghiệm: giảm trung bình 0.6795 pp khi bỏ, CI dương.
4. **Emotion auxiliary** và **uncertainty weighting** có point estimate thấp hơn full, nhưng với chỉ 3 seed, CI vẫn chạm 0; chưa nên gọi đây là khác biệt có ý nghĩa chắc chắn.
5. ECE của một số ablation thấp hơn full, đặc biệt `no_emotion_auxiliary`. Điều này cho thấy có trade-off giữa calibration và F1; không nên chọn ablation chỉ vì ECE thấp hơn.

Kết quả trên là bằng chứng ablation có kiểm soát, không phải tuyên bố nhân quả tuyệt đối. Cỡ mẫu seed `n=3` còn nhỏ nên các kết luận về thứ hạng sát nhau cần được giữ ở mức thận trọng.

## 6. Kiểm tra tính toàn vẹn khoa học

- Không dùng test set trong quá trình training.
- Checkpoint và threshold được freeze trước khi chạy test.
- Threshold được chọn trên dev.
- Giữ toàn bộ seed có cấu trúc hợp lệ; không lọc theo test outcome.
- Expected metric ranges chỉ là target để diễn giải, không phải điều kiện loại run.
- Full reference lấy từ source V37 đã freeze; không trộn với V22.
- Priority 3 được bỏ qua theo scope đã định; báo cáo này chỉ kết luận cho Priority 1.

## 7. Artifact và khả năng tái lập

- Config chính: `configs/experiments/priority12_xlmr.yaml`
- Aggregate: `reports/priority12_v37/aggregate_priority12.json`
- Run reports: `reports/priority12_v37/priority1/`
- Local run artifacts: `results/runs/priority12_v37_q1_*`
- HF campaign: `vipragsent-priority12-v37-20261003`
- HF upload queue/status: `runtime/priority12_v37_hf_upload_queue.jsonl`, `runtime/priority12_v37_hf_uploader_status.json`

Artifact routing đã loại repository bị đánh dấu unusable và dùng các repository overflow hợp lệ. Audit cuối ghi nhận đủ 12 receipt PASS, 22 file mỗi run.

## 8. Verification đã chạy

Các kiểm tra sau đều thành công:

- `ruff check scripts/run_priority12_xlmr.py scripts/aggregate_priority12.py scripts/hf_artifact_uploader.py src/vipragsent/data/masks.py`
- `py_compile` cho runner, aggregator và uploader
- `pytest -q tests/test_data_contract.py tests/test_full_xlmr_runner_contract.py tests/test_training_engine.py`
- Kết quả test: **12 passed**; không có lỗi functional.

## 9. Kết luận và khuyến nghị

Priority 1 V37 hiện **đủ điều kiện được coi là một campaign hợp lệ và hoàn tất**. Kết quả headline nên dùng là:

> **Full V37 Q1a XLM-R Large: 93.6658% ± 0.1871 pp macro pragmatic F1 trên 3 seed.**

Khuyến nghị dùng full V37 làm cấu hình chính cho báo cáo/đánh giá tiếp theo. Trong phần phân tích thành phần, có thể nêu polarity auxiliary và rationale supervision là hai thành phần có bằng chứng đóng góp rõ nhất trong campaign này; emotion auxiliary và uncertainty weighting cần thêm seed hoặc lặp lại độc lập trước khi kết luận mạnh hơn.

## Phụ lục: run IDs

- Full reference: `priority12_v37_q1_full_20260521`, `priority12_v37_q1_full_20260522`, `priority12_v37_q1_full_20260523`
- No emotion: `priority12_v37_q1_no_emotion_auxiliary_20260521/22/23`
- No polarity: `priority12_v37_q1_no_polarity_auxiliary_20260521/22/23`
- No rationale: `priority12_v37_q1_no_rationale_20260521/22/23`
- No uncertainty: `priority12_v37_q1_no_uncertainty_weighting_20260521/22/23`
