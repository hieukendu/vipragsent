# 65 Azure content-policy samples for manual rationale generation

> Generated from the frozen ViPragSent input and the authoritative failure report on 2026-08-08T00:15:33.092016+00:00.
> This file is a manual/offline handoff only. It made no Azure requests and does not change or promote experiment artifacts.

## Important use instructions

- There are exactly 65 samples below; keep every `sample_id` and comment unchanged.
- To preserve the original rationale-only semantics, send only the fenced `Model input` comment to GPT-4.1-mini.
- The gold labels and Azure filter reason are reference metadata. Do not include them in the model prompt unless you intentionally want label leakage.
- Do not treat generated rationales as canonical or approved. They must still pass the repository schema, provenance, audit, and approval gates.
- Do not retry the original Azure requests from this file without an explicit protocol decision.

## Frozen generation prompt

```text
Generate a rationale for this Vietnamese comment:
{comment}
```

The repository rationale schema requires a rationale string wrapped as `<RATIONALE>...</RATIONALE>`, together with the unchanged sample ID, comment, and source input hash. A suggested output envelope is:

```json
{"sample_id":"<sample_id>","comment":"<unchanged comment>","rationale_target":"<RATIONALE>...</RATIONALE>","source_input_hash":"<source_input_hash>"}
```

## Label summary for these 65 samples

| Label/category | Count |
|---|---:|
| `hate` | 50 |
| `jailbreak/prompt_shield` | 2 |
| `other` | 0 |
| `self_harm` | 2 |
| `sexual` | 0 |
| `unspecified_content_filter` | 4 |
| `violence` | 7 |

The table above is Azure filter-category metadata. Gold-label distributions are included per sample below.

## Samples

### 1. `fresh_visobert_00009`

**Model input**

```text
Có khi nào một trong những ng con bị đuổi đi đã thuê sát thủ tới giết cả nhà để hưởng trọn mảnh đất ko mn? triệu đô thời đó mà chia cho 7 người thì nó ít, mà chia cho người thì nhiều. Thậm chí đến nay triệu đô vẫn là nhiều vãi, và mảnh đất đó tới nay chắc cũng ko còn giá đó đâu 🤷🏻♀
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "disgust", "idiom_figurative": 1, "implicit_sentiment": 1, "irony": 1, "mocking": 1, "polarity": "neutral", "sarcasm": 1}`
- Source input hash: `F33EEAB378109BD324B678EE01AAFC1565340B0C503765F8942E17EA575BE15C`
- Azure reason group: `azure_content_filter|violence:medium`
- Recorded filter categories: `[{"category": "violence", "detected": null, "filtered": true, "severity": "medium"}]`

### 2. `fresh_visobert_00032`

**Model input**

```text
Đương nhiên k phai Đông Phương ca rôi 😂😂😂 khi phach như ông phai thân hoa phuc, cô ao đê phanh, cươi dâm ta 😂😂😂😂😂😂
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "enjoyment", "idiom_figurative": 1, "implicit_sentiment": 1, "irony": 0, "mocking": 1, "polarity": "positive", "sarcasm": 1}`
- Source input hash: `C1C6F05A5EB3F48FC62A6694A9AB4AEF6887A6D128E5B27E5B3682F41D6989A1`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
### 3. `fresh_visobert_00117`

**Model input**

```text
Áp lực gì,? Ai o phải mưu sinh,, biện hộ cho sự ác độc của mình,, o nuôi dạy con tử tế thì gửi vào chùa,, hy vọng cháu bé sẽ phục hồi sức khỏe,, và may mắn o bị bạo hành nữa,, sống trong gia đình thừờng xuyên bị bạo hành,, sẽ rất khó là người phụ nữ dịu dàng sau này,, và sẽ bạo hành con cái tương lai của cháu,, lấy vợ xem tông là vậy,, ý kiến cá nhân của mình,,
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "other", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `F0F94A98226D72F87F54CEDBE5EA77EDC0C79F5045B50051B1B34FAFEFAF63BA`
- Azure reason group: `azure_content_filter|violence:medium`
- Recorded filter categories: `[{"category": "violence", "detected": null, "filtered": true, "severity": "medium"}]`

### 4. `fresh_visobert_00135`

**Model input**

```text
mấy bắc kì cặc bọn bâu vô phân biệt
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "other", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `FED856AF3135F7E355B78B3126966C2A3D6B59B105EB1DDBA234A97F94C3A854`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 5. `fresh_visobert_00515`

**Model input**

```text
hahaha đĩ nl con cặc sinh cl sình
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "enjoyment", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `5BED94E609BC4FDCCA95F6261F8AED8818446C497519347826C8A3D35D310CE0`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 6. `fresh_visobert_00747`

**Model input**

```text
xuất cai l đm quần áo vn toàn nhập siêu cặc tq 7 xuất xuất clm chúng mày
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "anger", "idiom_figurative": 1, "implicit_sentiment": 1, "irony": 0, "mocking": 1, "polarity": "negative", "sarcasm": 0}`
- Source input hash: `9709C29D4F233C9A22256F12533324C578CD1EA79FA0DB4D99D831B35A398D2D`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 7. `fresh_visobert_01029`

**Model input**

```text
9 4 chửi chửi lgbt
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "other", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `E4B4B89F3CD2DA14A1A1E8F25099E598B34D37D73A44A8E898C80091E077A9F3`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 8. `fresh_visobert_01087`

**Model input**

```text
dân ngu cu cặc đen chết
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "disgust", "idiom_figurative": 0, "implicit_sentiment": 1, "irony": 0, "mocking": 1, "polarity": "negative", "sarcasm": 0}`
- Source input hash: `44888C32309F4EDA66FD0EC9CABEA7575F90806EC833FC7D9C026783D961C8A9`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 9. `fresh_visobert_01185`

**Model input**

```text
dẹp lũ sư quốc doanh phá hoại đạo cặc pháp cơ sở bọn nằm vùng phá nát văn hóa vn
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "other", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `8577F3CBF77ADDEDFF8FD6AD1DDA5462E1768D06C38CC263A4B478EFC13C5C31`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 10. `fresh_visobert_01280`

**Model input**

```text
ba que bọn chốc chỉ mị dân cai trị tham nhủng đất nước tụt hậu thầy bọn cặc bò đỏ bò mãi nhồi sọ umê chửi bò nghỉ đừng nội tạng giặc tàu
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "other", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `D5C7D53E906117BEAD8EEBFBF7348E1FBA99880D72D2EEC0B8EABF8EA62CD22A`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 11. `fresh_visobert_01315`

**Model input**

```text
dm bọn cặc việt tân
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "anger", "idiom_figurative": 0, "implicit_sentiment": 1, "irony": 0, "mocking": 1, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `7C147D44CD342B3C6E19FEDBCFC00930ED7F06E7B40B2EE37E203FAD806FCDF8`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 12. `fresh_visobert_01621`

**Model input**

```text
mấy cặc bệnh hoạn
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "disgust", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "negative", "sarcasm": 0}`
- Source input hash: `96B0550F90C68CF3623579699FD8521F531AC9B5F9A3D14BB6F6D6E456678916`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 13. `fresh_visobert_01826`

**Model input**

```text
chứ mấy bụng cặc bầu rước giặc diệt chủng đồng bào
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "other", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `67A0D31E000CFA8959AD0E163C897FA7F6DE20713803A60409D2A9F765F20728`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 14. `fresh_visobert_01849`

**Model input**

```text
mấy ngu dốt cặc gọi bọn nhà trí thức
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "disgust", "idiom_figurative": 0, "implicit_sentiment": 1, "irony": 0, "mocking": 1, "polarity": "negative", "sarcasm": 0}`
- Source input hash: `6588779A1D4EF09C134556AD089A8291422A0BAADCB84F0BF718792731685DF1`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 15. `fresh_visobert_02461`

**Model input**

```text
đm bọn cặc trung quốc chó
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "anger", "idiom_figurative": 0, "implicit_sentiment": 1, "irony": 0, "mocking": 1, "polarity": "negative", "sarcasm": 0}`
- Source input hash: `AC5213C86D30A6F31E07EB423A206080D803242F15EF9DF3BF24366A1611CD72`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 16. `fresh_visobert_02766`

**Model input**

```text
khuyết tật vé số ăn xin khuyết tật cặc đầu ngoẹo làm quan thái thú cai trị dân tộc
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "other", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `3E1DDFD6EB737E6F68304DCDC3118111A98EDB1B5CDF11833DAC1DE720937000`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 17. `fresh_visobert_03203`

**Model input**

```text
bọn sán dây tuyên cặc giáo rả rời chân lủ
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "other", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `EF0CF8C6020C1C46B1812FCACD04AC898D8C3D14A5B384D0ADCF76D671C536B2`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 18. `fresh_visobert_03221`

**Model input**

```text
dân ngu cu cặc đen
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "disgust", "idiom_figurative": 0, "implicit_sentiment": 1, "irony": 0, "mocking": 1, "polarity": "negative", "sarcasm": 0}`
- Source input hash: `1C873E55C26B1762C9CC987C275A88784335E9240B2966D33D91B9DFCED70D34`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 19. `fresh_visobert_03343`

**Model input**

```text
xàm lồn chửi chửi cặc chủ yếu ăn tiền mạng con khùng thúi
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "disgust", "idiom_figurative": 0, "implicit_sentiment": 1, "irony": 0, "mocking": 1, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `4D0B6B6FD9215ADBB5D2E21DFFFCCE29F7B17F7C2B5787B594FB99222F47E363`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 20. `fresh_visobert_03349`

**Model input**

```text
phản động khắp mấy cặc bán nước hút máu dân đen cứ sủa luật bọn chó
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "other", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `93429DA9C8D3212F5354725EF697B136D0D481E0DA16BB4A96C1C634398F1437`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 21. `fresh_visobert_03559`

**Model input**

```text
chửi lắm lgbt ❤ 🌈 nè chúc khỏe vui vẻ
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "enjoyment", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "positive", "sarcasm": 0}`
- Source input hash: `6DF1E0F71E9E3233E0D15AA4295FEC15E4372948EC59719931505B506737C1D4`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 22. `fresh_visobert_03794`

**Model input**

```text
ta ước gì lão hitler tái sinh cặc đất nước sạch lũ cặn bã dơ bẩn
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "disgust", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `09C876CB3C55391A753B9C6683680F5CCE6238993319E38982B0FEC52217E410`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 23. `fresh_visobert_03797`

**Model input**

```text
mọc trang cặc viẹt tân ag đkm bọn chó đẻ
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "other", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `88BA2AF9552709306DEB767B7AA433FF4A50800DB302C96B53254516DDC7C13D`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 24. `fresh_visobert_03878`

**Model input**

```text
tinh trùng khuyết tật cặc xaolol mua xài
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "other", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `234CDEB72734F195198AD53E4E75567681F79F412F6F107839106998BBAAD3A9`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 25. `fresh_visobert_04048`

**Model input**

```text
nhốt thạ chạy lung tinh 😏 😏
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "surprise", "idiom_figurative": 0, "implicit_sentiment": 1, "irony": 0, "mocking": 0, "polarity": "neutral", "sarcasm": 1}`
- Source input hash: `30D0ECF90280CB38745310AF6E4C87706098FEE81F38878E94CB46FE43028A7B`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 26. `fresh_visobert_04524`

**Model input**

```text
supi trần hự hự cắt tâm hồn
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "other", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `FA494256AD0386C5D08139EDD0BD09C336FF696F7DA205E1D9E1D20F3F4F4AB2`
- Azure reason group: `azure_content_filter|self_harm:medium`
- Recorded filter categories: `[{"category": "self_harm", "detected": null, "filtered": true, "severity": "medium"}]`

### 27. `fresh_visobert_04730`

**Model input**

```text
sủa chó đc củ đ cơm bọn kia cặc đ đc định cư kiếp bò
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "other", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `03E773E8B4BF8C579209A14E2F1DB42B0893D639EBFC9DDA03D5E8466EADC7B9`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 28. `fresh_visobert_04787`

**Model input**

```text
đẹp anh chị =)))
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "enjoyment", "idiom_figurative": 0, "implicit_sentiment": 1, "irony": 0, "mocking": 1, "polarity": "positive", "sarcasm": 1}`
- Source input hash: `030C37DE7775CFF7373FB6AFF94D485F9E2D8B10215FAE7A580E10E5DD164B6B`
- Azure reason group: `client_content_filter_message|category_not_recorded`
- Recorded filter categories: `[]`

### 29. `fresh_visobert_04911`

**Model input**

```text
đông lào tận chung kết quơn cặc mong liều mocphin lũ cừu cáo ngơ
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "other", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `33FFD6FED78F97785E21FF3B37839A2795EC8098AE868275EF52C70272431403`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 30. `fresh_visobert_05040`

**Model input**

```text
linh hun tật
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "other", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `3AD732C604C7F08AD73E261A481EC0C0F890DD2AFF2CA4361E008E2A65E63140`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 31. `fresh_visobert_05086`

**Model input**

```text
địt cụ chúng nó việt nam giáo sư tiến sỹ thế giới cặc chẳng lồn chất độc bọn bú lồn ăn tục nói phét
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "disgust", "idiom_figurative": 0, "implicit_sentiment": 1, "irony": 0, "mocking": 1, "polarity": "positive", "sarcasm": 0}`
- Source input hash: `B25527AF4A770321DD46EBC2482B023280877CD4F9A4415815BD83269EDBF49F`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 32. `fresh_visobert_05188`

**Model input**

```text
dkm chửi bọn cặc chó vàng made in vietnam
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "other", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `71DC0BBD6AE5AD4CE6F022E2294C622779D432D250CC85B90DF71EF1FAF789D0`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 33. `fresh_visobert_05276`

**Model input**

```text
ghét trung cộng hình như cặc ta múa quyền phù cáo mùi tàu khựa
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "sadness", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "negative", "sarcasm": 0}`
- Source input hash: `362C39A668B18C5C87DC5B14E9D0D5DCE007F23B60346B9FD360D41F77513C73`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 34. `fresh_visobert_05369`

**Model input**

```text
việt ngu cặc a dua trung cẩu
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "disgust", "idiom_figurative": 0, "implicit_sentiment": 1, "irony": 0, "mocking": 1, "polarity": "negative", "sarcasm": 0}`
- Source input hash: `0119BC45D1D42C8F827943DC8D1465737769E9EA5E8A9C3BC4B4A0B1354C3FA2`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 35. `fresh_visobert_05665`

**Model input**

```text
tr 4 xạo ngu to mặt dạy đời cặc hàng dễ thương ))) mồm mày gai tre tối xộn lào ak
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "enjoyment", "idiom_figurative": 0, "implicit_sentiment": 1, "irony": 0, "mocking": 1, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `C2D140C7BEFF36BC9FAF575AC1153933FF69381A63EB25BFCAAE167A29E8EFB0`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 36. `fresh_visobert_05698`

**Model input**

```text
báo cáo tài chính hợp nhất cặc mày óc chó ntn lũ thổ dân mồm vẩu
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "disgust", "idiom_figurative": 0, "implicit_sentiment": 1, "irony": 0, "mocking": 1, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `28407D22EBF4D5C269EB307FA09E2621A66C4398AC9BD7ED678AC96A11C7C8C2`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 37. `fresh_visobert_05765`

**Model input**

```text
linh oi mày vn luon nhe
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "other", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `8CDD409412ECBA0310B70ED9E9B512A81FA8FCAFA9E44D2C887868E0873947E6`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 38. `fresh_visobert_06140`

**Model input**

```text
Vụ khai sáng hôm sau bị cô gái đòi ,5 triệu đồng, hai bên đã to tiếng. Vụ không đồng ý nên đánh, siết cổ nạn nhân khiến tử vong.
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "other", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `4A6C003C0C50CE041F64CE5F2278851B322DB480CB7EF204966E730CA9CBBE02`
- Azure reason group: `azure_content_filter|violence:medium`
- Recorded filter categories: `[{"category": "violence", "detected": null, "filtered": true, "severity": "medium"}]`

### 39. `fresh_visobert_06325`

**Model input**

```text
đúng kiểu đến mùa động dục . khác gì động vật bậc thấp không
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "enjoyment", "idiom_figurative": 0, "implicit_sentiment": 1, "irony": 0, "mocking": 0, "polarity": "negative", "sarcasm": 0}`
- Source input hash: `653B67F91A77CBF8EE8C07689CF3B01ED253F9FD34E707C70A2F42F3E8B60CBB`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 40. `fresh_visobert_06928`

**Model input**

```text
đổi cả cuộc đời giống người tàn tật để được cái video này .
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 1, "emotion": "other", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `F8BF89DA47FCB61BC3A39F5CC0B3242CFC58AE9F1DD3B2DD42FD09A378BED369`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 41. `fresh_visobert_07142`

**Model input**

```text
Chào e gái xinh đẹp
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "enjoyment", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "positive", "sarcasm": 0}`
- Source input hash: `E9D20126BCC0F579042423265A43ADEED988DE4DA30F712949883C9EA86F99A2`
- Azure reason group: `client_content_filter_message|category_not_recorded`
- Recorded filter categories: `[]`

### 42. `fresh_visobert_07214`

**Model input**

```text
Con Meo nhà e sắp thành lợn rồi chị ơi =)))
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "disgust", "idiom_figurative": 0, "implicit_sentiment": 1, "irony": 0, "mocking": 1, "polarity": "neutral", "sarcasm": 1}`
- Source input hash: `916639A3CD47365B96BBC6835919FF6C5AC364E699BB0AE8446112EEE98B5517`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 43. `fresh_visobert_07278`

**Model input**

```text
Cả đều là loại chẳng ra gì, hợp vồn, nên chém hết cả cho đi nguyên cặp :v Giải thoát các nạn nhân kia :v
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "other", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `50A7712C70D8E7A271DC300F35A029C86B060085AEB7497B163E97BB142BD924`
- Azure reason group: `azure_content_filter|violence:medium`
- Recorded filter categories: `[{"category": "violence", "detected": null, "filtered": true, "severity": "medium"}]`

### 44. `fresh_visobert_07731`

**Model input**

```text
e gái loli và alime
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "other", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `8AE21A2418C057F1E5DD262E57B7707B4900A351D2DEB83A2613BBD28CCD15E4`
- Azure reason group: `client_content_filter_message|category_not_recorded`
- Recorded filter categories: `[]`

### 45. `fresh_visobert_07823`

**Model input**

```text
Lắm con nhỏ thế
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "enjoyment", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "positive", "sarcasm": 0}`
- Source input hash: `B2BB794D8DBBCF590FA2F27258889F3488199DFA1660630334068A3E51445D5F`
- Azure reason group: `client_content_filter_message|category_not_recorded`
- Recorded filter categories: `[]`

### 46. `fresh_visobert_08076`

**Model input**

```text
Chin-ya Lu 😂😂
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "enjoyment", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "positive", "sarcasm": 0}`
- Source input hash: `8041F00C50B72ACBF4FAE1F8AA5169D36746C4E300EDB402635142718041BD27`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 47. `fresh_visobert_08581`

**Model input**

```text
My Tây😂😂😂dat zo thui nhe.k dat ra
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 1, "emotion": "enjoyment", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `DB7B2126663C9C89DFD43F5FF541B60FDFF00535FDE3B1908555A50A66974FC6`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 48. `fresh_visobert_08734`

**Model input**

```text
Có hố thì tự nhảy xuống tự lấp lại dùm luôn đi nè hihi 😂
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "enjoyment", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "positive", "sarcasm": 0}`
- Source input hash: `D2F333E0BC5A3EC276799E5935E34FA613758C85D6DD31B8AC09874A9340D253`
- Azure reason group: `azure_content_filter|self_harm:medium`
- Recorded filter categories: `[{"category": "self_harm", "detected": null, "filtered": true, "severity": "medium"}]`

### 49. `fresh_visobert_08823`

**Model input**

```text
Nguyễn Anh Phước t chửn bị nửa lít axit, chặt 1 cái t tạt dô tay liềnnnn
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "other", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `C83EFAAA8B68D0DFD0D3E56CD45357AE771CB777D18BA52F3A63F6A76AC653EE`
- Azure reason group: `azure_content_filter|violence:medium`
- Recorded filter categories: `[{"category": "violence", "detected": null, "filtered": true, "severity": "medium"}]`

### 50. `fresh_visobert_09082`

**Model input**

```text
Cứ động đến trẻ con là t đã muốn đấm cho con mẹ này chục phát rồi
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "other", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `BE82024C60CEE22D76301A4227E7ED1443F1426FDBA0221228A19752426C24B3`
- Azure reason group: `azure_content_filter|violence:medium`
- Recorded filter categories: `[{"category": "violence", "detected": null, "filtered": true, "severity": "medium"}]`

### 51. `fresh_visobert_09135`

**Model input**

```text
Nhung béo và mẹ thắng là hh chuyển giới xịn đấy
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "enjoyment", "idiom_figurative": 0, "implicit_sentiment": 1, "irony": 0, "mocking": 0, "polarity": "positive", "sarcasm": 0}`
- Source input hash: `E81DA5527A3809C7C9812FD1EC98F0277F158634737C68D7C1E306A47769DA12`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 52. `fresh_visobert_09326`

**Model input**

```text
Trần Thị Tuyết Trinh co kham tu cung co kham Bv luon e chac cu goy e oi kkkk
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "other", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `FADF797D6327726B202B59EA677D821DF7D6ABC061FB6D4C5C3A7FFC22983EA3`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 53. `fresh_visobert_09813`

**Model input**

```text
E là fan âm mộ của ckj nèk
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 1, "emotion": "other", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `D4DD6EA829EB81D0086938679F9F47D0EA91A44236978A0375CCD049A65F92A6`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 54. `fresh_visobert_09853`

**Model input**

```text
Đen thôi,đo quên đi
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "other", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `336A75B38BA66AD9766F003591D6D42DA33C1D73F3679C433B072456FE404337`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 55. `fresh_visobert_10009`

**Model input**

```text
T k kì thị người tật nguyền nhưng riêng với thằng này thì nó xứng đáng =)))))
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "disgust", "idiom_figurative": 0, "implicit_sentiment": 1, "irony": 1, "mocking": 1, "polarity": "neutral", "sarcasm": 1}`
- Source input hash: `621F39C56309215AC501A42202D56E024BF3DDBB4146C07091DC1F4668E2332C`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 56. `fresh_visobert_10226`

**Model input**

```text
Mấy thằng xạo quần này nước tao đầy
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "other", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `21919499F42DB00D224AAEA950D274238243BA484AFFB64EB24FEBB2470C4F17`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 57. `fresh_visobert_10421`

**Model input**

```text
Nên bắt dân Việt Nam! Cắt 1 kiểu đầu của chủ tịch thôi
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "sadness", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "positive", "sarcasm": 0}`
- Source input hash: `1AEB952806A71364407D66F0F764C24E8C195555B3F9E3C2B67AD25BBDD68AF7`
- Azure reason group: `azure_content_filter|violence:medium`
- Recorded filter categories: `[{"category": "violence", "detected": null, "filtered": true, "severity": "medium"}]`

### 58. `fresh_visobert_10570`

**Model input**

```text
Goy goy 🙂 e bị ma theo á 🙂
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "surprise", "idiom_figurative": 0, "implicit_sentiment": 1, "irony": 0, "mocking": 0, "polarity": "neutral", "sarcasm": 1}`
- Source input hash: `BE9A4829C1AB5458B3517B0C79CE66FB669AC8559BFCB003B719DA10F9249D51`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 59. `fresh_visobert_10577`

**Model input**

```text
Đúng rồi.Chứ k phải do mấy thằng bụng bầu rước giặc về diệt chủng đồng bào đâu
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "other", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `4692A18CC9E2D65D56BA595E8FCEE2DAA3E27C28BB9DE1E8A8D178E300F6C1E7`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 60. `fresh_visobert_10825`

**Model input**

```text
Bảo Phan Homo Sapiens là giống loài dị tật =))
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "other", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `F257FD45999048DFCC869CA4AD4A71B8F11DBD4F5A15F522321AB985ED22464F`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 61. `fresh_visobert_11218`

**Model input**

```text
chảnh chóa, tự kỉ, giống ma nhập
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "other", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `72E493240B3CB7B9D7CBCD293FAF3C21EB150607F986E185643C465B15A572CA`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 62. `fresh_visobert_11490`

**Model input**

```text
izizaatsyzix kv visak l odzub. Lizrv blhxuf,u gk. yxrbbv gxzgx. Phzhx bhciszoc. Loixlb
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "other", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `213570D3B34801F2900B09FB8F3BBF1AE0AC005D581A33F043D691268CC54903`
- Azure reason group: `azure_content_filter|jailbreak:severity_not_recorded`
- Recorded filter categories: `[{"category": "jailbreak", "detected": true, "filtered": true, "severity": null}]`

### 63. `fresh_visobert_11580`

**Model input**

```text
Mọe ra cơm bụi k cơm như cái ĐB, đã thế thịt còn hoy hoy nữa chứ. Con mọe nó
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "other", "idiom_figurative": 1, "implicit_sentiment": 1, "irony": 0, "mocking": 0, "polarity": "neutral", "sarcasm": 0}`
- Source input hash: `9E9B8D487BE3B1E9221D48519A67520364D0AC1EA71B1C7008725347FB8EC88D`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`

### 64. `fresh_visobert_11708`

**Model input**

```text
fdp zew gjg. ppo ycr kzi, srh kbw uig xlb png een sqf jyx rmnn mff. okd ylj uca lxs bwo zjf aez xlu flb wqs . wxx cja dqo ciz tga, jtx uas had jys gbn , w]gl bxv jps wvs jvg hfx tmf hsx xlv jqr .
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 1, "emotion": "enjoyment", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "positive", "sarcasm": 0}`
- Source input hash: `8A70031CA529E186B08DEA6725FC8471F8CF99D313CCA28257E18938C7FB37B4`
- Azure reason group: `azure_content_filter|jailbreak:severity_not_recorded`
- Recorded filter categories: `[{"category": "jailbreak", "detected": true, "filtered": true, "severity": null}]`

### 65. `fresh_visobert_11853`

**Model input**

```text
đã không sợ mất thì còn lên đây lăn tăn làm gì, về lại sao hỏa đi má
```

**Reference metadata — omit from the model prompt for rationale-only generation**

- Gold labels: `{"code_switching": 0, "emotion": "fear", "idiom_figurative": 0, "implicit_sentiment": 0, "irony": 0, "mocking": 0, "polarity": "negative", "sarcasm": 0}`
- Source input hash: `328673F1602FB0840866EDECC22B95D719C42AA35D92A1EBF94FD5F869449F90`
- Azure reason group: `azure_content_filter|hate:medium`
- Recorded filter categories: `[{"category": "hate", "detected": null, "filtered": true, "severity": "medium"}]`
