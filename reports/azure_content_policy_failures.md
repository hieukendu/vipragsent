# Azure content-policy forensic audit

Status: `PIPELINE_SEMANTICS_AMBIGUOUS`

This is a read-only audit of exactly 65 failed `azure_rationale_generation` samples. It made zero Azure requests, did not retry any sample, did not change the frozen protocol, and did not promote or start downstream work.

## Conclusion

All 65 failure rows are attributable to Azure content-policy handling in the retained evidence: 61 preserve a structured HTTP 400 `content_filter` / `ContentFiltered` prompt rejection, and 4 preserve the repository's terminal `Azure response was terminated by content filtering` exception without the provider payload. The four generic rows have no retained category, severity, or provider `source_type`; those fields are intentionally reported as unknown.

The repository has a run-level stop/review gate, but no authoritative sample-level rule for retrying, falling back, marking unavailable, excluding, or regenerating these rationale samples. The result is `PIPELINE_SEMANTICS_AMBIGUOUS`; the incomplete 7,933-row artifact remains unpromoted.

## Category counts

| Category | Count |
|---|---:|
| `sexual` | 0 |
| `violence` | 7 |
| `hate` | 50 |
| `self_harm` | 2 |
| `jailbreak/prompt_shield` | 2 |
| `unspecified_content_filter` | 4 |
| `other` | 0 |

`jailbreak/prompt_shield` is counted from the provider's recorded `jailbreak` field; no `prompt_shield` field was present. Category counts are based only on provider payload fields and the four generic rows are counted as `unspecified_content_filter`.

## Exact reason groups

| Exact reason group | Count | Sample IDs |
|---|---:|---|
| `azure_content_filter|hate:medium` | 50 | `fresh_visobert_00032, fresh_visobert_00135, fresh_visobert_00515, fresh_visobert_00747, fresh_visobert_01029, fresh_visobert_01087, fresh_visobert_01185, fresh_visobert_01280, fresh_visobert_01315, fresh_visobert_01621, fresh_visobert_01826, fresh_visobert_01849, fresh_visobert_02461, fresh_visobert_02766, fresh_visobert_03203, fresh_visobert_03221, fresh_visobert_03343, fresh_visobert_03349, fresh_visobert_03559, fresh_visobert_03794, fresh_visobert_03797, fresh_visobert_03878, fresh_visobert_04048, fresh_visobert_04730, fresh_visobert_04911, fresh_visobert_05040, fresh_visobert_05086, fresh_visobert_05188, fresh_visobert_05276, fresh_visobert_05369, fresh_visobert_05665, fresh_visobert_05698, fresh_visobert_05765, fresh_visobert_06325, fresh_visobert_06928, fresh_visobert_07214, fresh_visobert_08076, fresh_visobert_08581, fresh_visobert_09135, fresh_visobert_09326, fresh_visobert_09813, fresh_visobert_09853, fresh_visobert_10009, fresh_visobert_10226, fresh_visobert_10570, fresh_visobert_10577, fresh_visobert_10825, fresh_visobert_11218, fresh_visobert_11580, fresh_visobert_11853` |
| `azure_content_filter|jailbreak:severity_not_recorded` | 2 | `fresh_visobert_11490, fresh_visobert_11708` |
| `azure_content_filter|self_harm:medium` | 2 | `fresh_visobert_04524, fresh_visobert_08734` |
| `azure_content_filter|violence:medium` | 7 | `fresh_visobert_00009, fresh_visobert_00117, fresh_visobert_06140, fresh_visobert_07278, fresh_visobert_08823, fresh_visobert_09082, fresh_visobert_10421` |
| `client_content_filter_message|category_not_recorded` | 4 | `fresh_visobert_04787, fresh_visobert_07142, fresh_visobert_07731, fresh_visobert_07823` |

## Failure verification

- Frozen input rows: `7998`; failed IDs: `65`; failed IDs present in frozen input: `True`.
- Successful usage records: `7933`; failed IDs with usage records: `0`.
- Failed IDs with rationale rows: `0`.
- Failed IDs with recomputed cache entries: `0`.
- Non-policy/parser/schema/timeout/transport error rows: `0` in the retained report; the four generic rows are content-filter exceptions whose provider details were not persisted.

## Existing protocol and repository policy

The authoritative run-level rule is to stop and keep review/approval pending: `configs/execution_policy.yaml:2-9` and `prompts/sequential/azure/azure_rationale_generation.md:3,18-33`. The Azure client treats content filtering as terminal (`src/vipragsent/azure/client.py:179-189,411-438`), the batch validator fails if any invalid/missing/failed response exists (`src/vipragsent/orchestration/stage_registry.py:1538-1549`), and promotion requires a completed, reviewed, explicitly approved run (`src/vipragsent/orchestration/rationale_promotion.py:87-104`).

No source specifies a sample-level choice A-E. Choice F exists only at the run level, not as a disposition for the 65 samples. The downstream all-zero invalid policy in `configs/experiments/generation_reasoning_protocol.yaml:40-44` is not applicable to missing Azure rationale artifacts.

## Per-sample forensic records

### `fresh_visobert_00009`
- Original Vietnamese comment: `Có khi nào một trong những ng con bị đuổi đi đã thuê sát thủ tới giết cả nhà để hưởng trọn mảnh đất ko mn? triệu đô thời đó mà chia cho 7 người thì nó ít, mà chia cho người thì nhiều. Thậm chí đến nay triệu đô vẫn là nhiều vãi, và mảnh đất đó tới nay chắc cũng ko còn giá đó đâu 🤷🏻♀`
- Reason group: `azure_content_filter|violence:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "violence", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/BBEFE3683AE9F943B4F3A35179BA0DD8A699087627D52FE18F372A6ACF11F2CF.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': False, 'severity': 'safe'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': True, 'severity': 'medium'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 590, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_00032`
- Original Vietnamese comment: `Đương nhiên k phai Đông Phương ca rôi 😂😂😂 khi phach như ông phai thân hoa phuc, cô ao đê phanh, cươi dâm ta 😂😂😂😂😂😂`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/10B6D34AAD2709258E31B8CECF2CF564B465D2FB5CA40F867AE3FE8A18768C00.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 422, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_00117`
- Original Vietnamese comment: `Áp lực gì,? Ai o phải mưu sinh,, biện hộ cho sự ác độc của mình,, o nuôi dạy con tử tế thì gửi vào chùa,, hy vọng cháu bé sẽ phục hồi sức khỏe,, và may mắn o bị bạo hành nữa,, sống trong gia đình thừờng xuyên bị bạo hành,, sẽ rất khó là người phụ nữ dịu dàng sau này,, và sẽ bạo hành con cái tương lai của cháu,, lấy vợ xem tông là vậy,, ý kiến cá nhân của mình,,`
- Reason group: `azure_content_filter|violence:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "violence", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/0DBBC5816C0DB1F40693B2D2288F7BB3592C969A8DF24618E71C1207ED58AFB8.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': False, 'severity': 'low'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': True, 'severity': 'medium'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 671, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_00135`
- Original Vietnamese comment: `mấy bắc kì cặc bọn bâu vô phân biệt`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/358F0EE84A99B83C70A77514AA17933027FB2AD20FA501F188C9A4720FC44F2A.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 343, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_00515`
- Original Vietnamese comment: `hahaha đĩ nl con cặc sinh cl sình`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/BA76973F705C63A2B408ECEB72FEB51BB86AFB49CFEB1AD5EA4952CAF0A6E5A3.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 341, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_00747`
- Original Vietnamese comment: `xuất cai l đm quần áo vn toàn nhập siêu cặc tq 7 xuất xuất clm chúng mày`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/B2375732AC408231047226DC4F56170C89FC16BA3CD90791FDAE59D5C0812689.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 380, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_01029`
- Original Vietnamese comment: `9 4 chửi chửi lgbt`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/E79D60E8906DC77A245606F95EF3E3A15661110FE2F414A125D89CFC5BC40076.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 326, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_01087`
- Original Vietnamese comment: `dân ngu cu cặc đen chết`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/7A26F0100DF9E647EE3971F53D6B74167376E2A2BAA8A78138C21B57985337CE.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 331, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_01185`
- Original Vietnamese comment: `dẹp lũ sư quốc doanh phá hoại đạo cặc pháp cơ sở bọn nằm vùng phá nát văn hóa vn`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/C782DA90CF6BBADFC149F41BF43E930E11487BD8129CCFEE28B97A2D914F6F90.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 388, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_01280`
- Original Vietnamese comment: `ba que bọn chốc chỉ mị dân cai trị tham nhủng đất nước tụt hậu thầy bọn cặc bò đỏ bò mãi nhồi sọ umê chửi bò nghỉ đừng nội tạng giặc tàu`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/3569E3E016CE5AC2E9DDD7D34D39ED4926F26B48D88266922F7CC2C22AC1B4F1.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 444, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_01315`
- Original Vietnamese comment: `dm bọn cặc việt tân`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/DB4A459CDCAE592E8D5DED89B28FBD8BED4EEFDACE4D6E54FC302058B446D42B.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 327, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_01621`
- Original Vietnamese comment: `mấy cặc bệnh hoạn`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/EE334367C574004B2292BE7BDDFDEFE3701F1849F5F5924697C3117B6680BE05.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 325, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_01826`
- Original Vietnamese comment: `chứ mấy bụng cặc bầu rước giặc diệt chủng đồng bào`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/17FBA4AAB73C5D4A264055441FF0DA4BFBC77991398339253CDC1DC082AFF270.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 358, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_01849`
- Original Vietnamese comment: `mấy ngu dốt cặc gọi bọn nhà trí thức`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/18CD31B693B4C85B96E8B8DC45D5A26C1D3602EB24BDC2B064071E00B506E795.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 344, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_02461`
- Original Vietnamese comment: `đm bọn cặc trung quốc chó`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/8E4E934A899C8C44D2541F15DBFFCEC3F469CA2F6F788AED4495EC35303B2B53.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 333, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_02766`
- Original Vietnamese comment: `khuyết tật vé số ăn xin khuyết tật cặc đầu ngoẹo làm quan thái thú cai trị dân tộc`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/306803BA3DB4F690A6A57A887325C1DA2854D5B9B839767533D280BA5B8AB447.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 390, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_03203`
- Original Vietnamese comment: `bọn sán dây tuyên cặc giáo rả rời chân lủ`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/CABAEFC480B6807F98AE3DAC3E10487A3F0AB366CC40896FFA36FAE7A6AD37F0.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 349, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_03221`
- Original Vietnamese comment: `dân ngu cu cặc đen`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/462CEC194E89446D1694174B4149B70387B29E06678663A990F844EFA36E4083.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 326, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_03343`
- Original Vietnamese comment: `xàm lồn chửi chửi cặc chủ yếu ăn tiền mạng con khùng thúi`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/B46F7B043F21BF8237C24DCE9FAF5439D2EAE486353CDA42B4307384DBED8F8A.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 365, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_03349`
- Original Vietnamese comment: `phản động khắp mấy cặc bán nước hút máu dân đen cứ sủa luật bọn chó`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/D0843D928F309E2B6A90601E5FB976C614AC1ED694B07025DE1BB7FDB7015667.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 375, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_03559`
- Original Vietnamese comment: `chửi lắm lgbt ❤ 🌈 nè chúc khỏe vui vẻ`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/F65A62631E745DE1B66D93492D0E025A6317A5DAB93133AFC290E85791B4E240.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 345, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_03794`
- Original Vietnamese comment: `ta ước gì lão hitler tái sinh cặc đất nước sạch lũ cặn bã dơ bẩn`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/B029D535E4BFA29F746724DA59EBE27E2266CBA7A6E97733B0D09F836F8703AE.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 372, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_03797`
- Original Vietnamese comment: `mọc trang cặc viẹt tân ag đkm bọn chó đẻ`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/3D2EA509F583215B39F370657D69EA8CD829D24AD60FAACDD7FFC017CB7D3DDF.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 348, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_03878`
- Original Vietnamese comment: `tinh trùng khuyết tật cặc xaolol mua xài`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/FD806F5D4118E9686F9A28D87CA645CF8E1114E47BC2119C85D1687D4141FED9.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 348, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_04048`
- Original Vietnamese comment: `nhốt thạ chạy lung tinh 😏 😏`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/5CDBA57B240A0CA0D2B331ED2AC8CE7CC28C282A021F564EB3490E5EE2E17C68.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 335, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_04524`
- Original Vietnamese comment: `supi trần hự hự cắt tâm hồn`
- Reason group: `azure_content_filter|self_harm:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "self_harm", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/1A561A60860A1526DFBAA85572A12E0EF2ECD8841E1610F852A99D65C63EF7BB.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': False, 'severity': 'safe'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': True, 'severity': 'medium'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 335, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_04730`
- Original Vietnamese comment: `sủa chó đc củ đ cơm bọn kia cặc đ đc định cư kiếp bò`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/BCC410B663DB30C29C292809962A34CF4659C009206F1F08B5288409500950AC.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 360, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_04787`
- Original Vietnamese comment: `đẹp anh chị =)))`
- Reason group: `client_content_filter_message|category_not_recorded`
- HTTP status / Azure code / inner code: `None` / `None` / `None`
- Content-filter source/categories: `None` / `[]`
- Inference reached: `None`; The persisted failure row retains only the client exception; provider payload/source_type was not persisted.
- Partial model-response artifact: `False`
- Token usage returned: `None`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/513341A7C52133BF83E70935F7D76ED1B8A6155A76B582C0CD182C733A18AAD7.json`)
- Exact persisted exception:
```text
Azure response was terminated by content filtering
```
### `fresh_visobert_04911`
- Original Vietnamese comment: `đông lào tận chung kết quơn cặc mong liều mocphin lũ cừu cáo ngơ`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/B17584D61591388FCD8453AC4BF5EAC3725AC65316A8BE53AE2DB3358D69E2EA.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 372, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_05040`
- Original Vietnamese comment: `linh hun tật`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/F8A8F181E3FE3C508C4E60259FE6E5D405BACAD2693F7D5AA27F9444F754898C.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 320, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_05086`
- Original Vietnamese comment: `địt cụ chúng nó việt nam giáo sư tiến sỹ thế giới cặc chẳng lồn chất độc bọn bú lồn ăn tục nói phét`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/79ED0677D96DC44C47945B0ACB43C7A66C45CC8D6CE6460B381C81C132D97A62.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 407, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_05188`
- Original Vietnamese comment: `dkm chửi bọn cặc chó vàng made in vietnam`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/0BEB8EEE2F73E507EE263F7715B50E00B34B66797249A4BD2C0B5DCE255C734E.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 349, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_05276`
- Original Vietnamese comment: `ghét trung cộng hình như cặc ta múa quyền phù cáo mùi tàu khựa`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/9440C653DF8703DF936AC3B7EE2658AF438C291CDA7FA24E1463A2BF60238369.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 370, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_05369`
- Original Vietnamese comment: `việt ngu cặc a dua trung cẩu`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/480B63AA2BF2B4B37C2706CD895E4014A4542D679A0D1B08CC6E3559A187ADCC.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 336, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_05665`
- Original Vietnamese comment: `tr 4 xạo ngu to mặt dạy đời cặc hàng dễ thương ))) mồm mày gai tre tối xộn lào ak`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/2C69887FF9EFDF5EB59FB01927884938B02EF68CA3AED4FAA7B3829765D38253.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 389, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_05698`
- Original Vietnamese comment: `báo cáo tài chính hợp nhất cặc mày óc chó ntn lũ thổ dân mồm vẩu`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/9C5EA683B562143FF309824084C14A4B9211D89A1B0C6EF8EBEB8C168BCF0C0B.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 372, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_05765`
- Original Vietnamese comment: `linh oi mày vn luon nhe`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/589548B526C59890CB31DE58B907CD0F75727784FF8095178DE39640CFCAC17B.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 331, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_06140`
- Original Vietnamese comment: `Vụ khai sáng hôm sau bị cô gái đòi ,5 triệu đồng, hai bên đã to tiếng. Vụ không đồng ý nên đánh, siết cổ nạn nhân khiến tử vong.`
- Reason group: `azure_content_filter|violence:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "violence", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/44A8D67BFBA2B5444A357CDF5DC57D680A183ED7DBC0A332DEE3EB2F2D5B9769.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': False, 'severity': 'safe'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': True, 'severity': 'medium'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 436, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_06325`
- Original Vietnamese comment: `đúng kiểu đến mùa động dục . khác gì động vật bậc thấp không`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/086CB7E0C4E3FC0970BD0F13E2CC00574BC56F1A1BA93C0ADF86D50A8482B7DE.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 368, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_06928`
- Original Vietnamese comment: `đổi cả cuộc đời giống người tàn tật để được cái video này .`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/DAFD92AE479BD55FB2077D571C2B481DFA8ADCC076F026F3D99A5D93C8D94E1B.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 367, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_07142`
- Original Vietnamese comment: `Chào e gái xinh đẹp`
- Reason group: `client_content_filter_message|category_not_recorded`
- HTTP status / Azure code / inner code: `None` / `None` / `None`
- Content-filter source/categories: `None` / `[]`
- Inference reached: `None`; The persisted failure row retains only the client exception; provider payload/source_type was not persisted.
- Partial model-response artifact: `False`
- Token usage returned: `None`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/730C24AC27A4363E0008B1B9DA4EF7D6C87B8F4694E6A0EB3C8B43777467A5FA.json`)
- Exact persisted exception:
```text
Azure response was terminated by content filtering
```
### `fresh_visobert_07214`
- Original Vietnamese comment: `Con Meo nhà e sắp thành lợn rồi chị ơi =)))`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/24B94FF6564F2E885D58CFFBB5BD475F1FE4C9B834C40CDF10BF0BB22B7DDEAE.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 351, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_07278`
- Original Vietnamese comment: `Cả đều là loại chẳng ra gì, hợp vồn, nên chém hết cả cho đi nguyên cặp :v Giải thoát các nạn nhân kia :v`
- Reason group: `azure_content_filter|violence:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "violence", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/8C6EEB7004E306656F68608C8C3283BD25811F3F12AA6E19C42218B3503BFAFD.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': False, 'severity': 'low'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': True, 'severity': 'medium'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 412, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_07731`
- Original Vietnamese comment: `e gái loli và alime`
- Reason group: `client_content_filter_message|category_not_recorded`
- HTTP status / Azure code / inner code: `None` / `None` / `None`
- Content-filter source/categories: `None` / `[]`
- Inference reached: `None`; The persisted failure row retains only the client exception; provider payload/source_type was not persisted.
- Partial model-response artifact: `False`
- Token usage returned: `None`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/612EDFFCB53E7C89AD05DE2C35218B903F167824B5E786E81F5356331BC4131B.json`)
- Exact persisted exception:
```text
Azure response was terminated by content filtering
```
### `fresh_visobert_07823`
- Original Vietnamese comment: `Lắm con nhỏ thế`
- Reason group: `client_content_filter_message|category_not_recorded`
- HTTP status / Azure code / inner code: `None` / `None` / `None`
- Content-filter source/categories: `None` / `[]`
- Inference reached: `None`; The persisted failure row retains only the client exception; provider payload/source_type was not persisted.
- Partial model-response artifact: `False`
- Token usage returned: `None`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/ABC21849F3D19D79B4B29DC54FDFB727131D784DBCEAF1B7DB430C89C03CA9CD.json`)
- Exact persisted exception:
```text
Azure response was terminated by content filtering
```
### `fresh_visobert_08076`
- Original Vietnamese comment: `Chin-ya Lu 😂😂`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/5FB7DC139BBDB8C702697705D5652C10114CBC69565D570AD72DD8305F67AA00.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 321, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_08581`
- Original Vietnamese comment: `My Tây😂😂😂dat zo thui nhe.k dat ra`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/2BDEFB1D15B2BCBA2CD58B2C883B36670830FE988BB4A32C413762F6618C622A.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 341, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_08734`
- Original Vietnamese comment: `Có hố thì tự nhảy xuống tự lấp lại dùm luôn đi nè hihi 😂`
- Reason group: `azure_content_filter|self_harm:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "self_harm", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/568F330FE1425B8CB01476453EAB50AA9D0446601B5C48530FD5F8CD643353B2.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': False, 'severity': 'safe'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': True, 'severity': 'medium'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 364, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_08823`
- Original Vietnamese comment: `Nguyễn Anh Phước t chửn bị nửa lít axit, chặt 1 cái t tạt dô tay liềnnnn`
- Reason group: `azure_content_filter|violence:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "violence", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/14A1103A97FB0A89A99E17AFE21411D3F4B71E6288455A8C6F946BACEA86985D.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': False, 'severity': 'safe'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': True, 'severity': 'medium'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 380, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_09082`
- Original Vietnamese comment: `Cứ động đến trẻ con là t đã muốn đấm cho con mẹ này chục phát rồi`
- Reason group: `azure_content_filter|violence:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "violence", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/4B6B42E732D84DDDB0D29C0E276BBD489ACEBACD544273C78880D2E03C9BB55A.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': False, 'severity': 'low'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': True, 'severity': 'medium'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 373, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_09135`
- Original Vietnamese comment: `Nhung béo và mẹ thắng là hh chuyển giới xịn đấy`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/7AD2B84D5B22A9C3C2FD8B8E4A1DCE0EA7582BB5C9A09220CA07E14A5D0B7BD3.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 355, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_09326`
- Original Vietnamese comment: `Trần Thị Tuyết Trinh co kham tu cung co kham Bv luon e chac cu goy e oi kkkk`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/3D97E8C5EEC2B21FEC44D555E62EF4C075C1A923AD33B7B2C8D341D35E3DA315.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 384, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_09813`
- Original Vietnamese comment: `E là fan âm mộ của ckj nèk`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/C34387CBCEEC844204EF81667711098316362B84E165E5001F5B17475FED6F4A.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 334, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_09853`
- Original Vietnamese comment: `Đen thôi,đo quên đi`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/994113CB259914079D6A2C556CD2453DF0E43B4F4A8747D1D5325165B2554776.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 327, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_10009`
- Original Vietnamese comment: `T k kì thị người tật nguyền nhưng riêng với thằng này thì nó xứng đáng =)))))`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/61BF7F6145E79A5D9A5B2553A84D7812D0251B86F0361328F8F193A4F31E9D19.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 385, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_10226`
- Original Vietnamese comment: `Mấy thằng xạo quần này nước tao đầy`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/744DB7B1E6B3512F7F716EED6B6E36674B9BEA4C13C2B8B2DEC5CF2371C1E286.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 343, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_10421`
- Original Vietnamese comment: `Nên bắt dân Việt Nam! Cắt 1 kiểu đầu của chủ tịch thôi`
- Reason group: `azure_content_filter|violence:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "violence", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/D7A88C947BC7F94507CAB57502C5AF97D9A9E73B3382FEABD773EAA0A57786AC.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': False, 'severity': 'safe'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': True, 'severity': 'medium'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 362, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_10570`
- Original Vietnamese comment: `Goy goy 🙂 e bị ma theo á 🙂`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/BE8EFEE3CA5EA6807640A8215A450100ED002EBAD3923459A4D695F1F18C9908.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 334, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_10577`
- Original Vietnamese comment: `Đúng rồi.Chứ k phải do mấy thằng bụng bầu rước giặc về diệt chủng đồng bào đâu`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/D70EEFD167723928702088BC92301DAC238775EC5E19D29CA8C7099A98C7C3CA.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 386, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_10825`
- Original Vietnamese comment: `Bảo Phan Homo Sapiens là giống loài dị tật =))`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/F10203E018BC7F33E526E7EB5A30405878924C59BA4457E3280E2DEFD98F0A95.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 354, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_11218`
- Original Vietnamese comment: `chảnh chóa, tự kỉ, giống ma nhập`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/A4F57549520939DC7366A4CFB9B20FE8733296A605D81165002A9BB7EDC7DCCD.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 340, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_11490`
- Original Vietnamese comment: `izizaatsyzix kv visak l odzub. Lizrv blhxuf,u gk. yxrbbv gxzgx. Phzhx bhciszoc. Loixlb`
- Reason group: `azure_content_filter|jailbreak:severity_not_recorded`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "jailbreak", "detected": true, "filtered": true, "severity": null}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/EF3D5EF3AB39B5DF084D2624AFB1C2376CD1F91A5B311AB12F4D94140AC25321.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': False, 'severity': 'safe'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': True, 'filtered': True}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 394, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_11580`
- Original Vietnamese comment: `Mọe ra cơm bụi k cơm như cái ĐB, đã thế thịt còn hoy hoy nữa chứ. Con mọe nó`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/FE0C5E15C7991F8099001517E91B0E36BD542F548C0F5CD8270B39CE62893F0E.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 384, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_11708`
- Original Vietnamese comment: `fdp zew gjg. ppo ycr kzi, srh kbw uig xlb png een sqf jyx rmnn mff. okd ylj uca lxs bwo zjf aez xlu flb wqs . wxx cja dqo ciz tga, jtx uas had jys gbn , w]gl bxv jps wvs jvg hfx tmf hsx xlv jqr .`
- Reason group: `azure_content_filter|jailbreak:severity_not_recorded`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "jailbreak", "detected": true, "filtered": true, "severity": null}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/0CB8251DDD0F2693DFEA04DBB8D1459DCE60055C090C109AED6B535FAE90277D.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': False, 'severity': 'safe'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': True, 'filtered': True}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 503, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
### `fresh_visobert_11853`
- Original Vietnamese comment: `đã không sợ mất thì còn lên đây lăn tăn làm gì, về lại sao hỏa đi má`
- Reason group: `azure_content_filter|hate:medium`
- HTTP status / Azure code / inner code: `400` / `content_filter` / `ContentFiltered`
- Content-filter source/categories: `prompt` / `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
- Inference reached: `False`; HTTP 400 content_filter with source_type=prompt indicates prompt-stage rejection before inference.
- Partial model-response artifact: `False`
- Token usage returned: `False`; usage record: `False`
- Cache entry: `False` (`results/runs/azure_rationale_generation/azure/cache/64AB83D107CD23544276662A94908FA2528233B181331FCB4C6FAD614A48FE35.json`)
- Exact persisted exception:
```text
Error code: 400 - {'error': {'message': 'The response was filtered due to the prompt triggering Azure OpenAI’s content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766', 'type': 'invalid_request_error', 'param': 'prompt', 'code': 'content_filter', 'content_filters': [{'blocked': True, 'source_type': 'prompt', 'content_filter_raw': [], 'content_filter_results': {'hate': {'filtered': True, 'severity': 'medium'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}}, 'content_filter_offsets': {'start_offset': 0, 'end_offset': 376, 'check_offset': 0}}], 'innererror': {'code': 'ContentFiltered'}}}
```
