# Review summarization evaluation · pilot

This is a **pretrained GPT-2** versus simple lead-16-words comparison. The original fine-tuned notebook checkpoint is unavailable and is not evaluated. The sample is 513 held-out rows from the notebook's seeded split, not an independently reviewed or deduplicated benchmark.

| Method | Examples scored | ROUGE-1 F1 | ROUGE-L F1 |
| --- | ---: | ---: | ---: |
| First 16 review words | 513 | 0.1069 | 0.0970 |
| Public pretrained GPT-2 | 513 | 0.0512 | 0.0493 |

Scores are macro-averages of per-example token-overlap F1 (lowercase alphanumeric tokens, no stemming). The lead baseline outperformed the untuned model in this sample. This is not a quality claim about the missing fine-tuned checkpoint, and overlap scores alone do not establish factuality or usefulness.

## Example outputs

These examples were selected from cases with zero model ROUGE-L to illustrate what the model produced, not as a representative error taxonomy. The reviews are preprocessed, so punctuation and stopwords differ from natural text. A zero-overlap score does not by itself prove an answer is wrong.

**Row 13705** — Review excerpt: “extremely high quality cat food tastes ( according cats ) good looks . seven ( yes , seven ) cats , rescues become spoiled brats enjoy difficult please . n't care fussie cat flavor…”<br>
Reference: “Fussie Cat is a hit with my "fussie cats."”<br>
GPT-2: “: good food, good food, good food, good food, good food, good food, good food, good food”

**Row 18671** — Review excerpt: “happy see product actually backed said flavor . purchased products stated `` bacon '' flavored dogs love . could n't get dogs eat ( one usually eats anything ) . problem getting do…”<br>
Reference: “My dog actually liked the flavor!”<br>
GPT-2: “: I'm not sure if this is a good idea or not. I'm not sure if this is a”

**Row 31176** — Review excerpt: “love chips . loved salt & vinegar potato chips decades best . 've ordered variety pack try offerings company .…”<br>
Reference: “Wow!!! These are great...”<br>
GPT-2: “: I love the variety pack. I love the variety pack. I love the variety pack. I love the variety pack”

## Method and limitations

- Source: repository dataset SHA-256 `7221e36e72c6d817dc8bc9ea30e8f274998dbb702fc82d8cc0fc6514ee76235c`; 25% test split with `random_state=42`; first 513 test rows in that shuffled order.
- Generation receives the review and ` TL;DR` only. References are supplied separately to scoring. Every row is evaluated, including the one-item final batch.
- Model: `openai-community/gpt2`, revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`; greedy decoding, 128 input tokens, up to 24 new tokens, batch size 4. The model has not been fine-tuned on this task in this evaluation.
- Generation took 185.337 seconds on the local CPU run; timing is hardware-dependent and excludes model loading, baseline scoring and data loading.
- Review titles are imperfect references. The split is not deduplicated or independently audited, and we have not run a human evaluation or confidence intervals.
- [Raw results and examples](pilot.json) record every evaluated row and configuration. See the [README](../README.md) for the reproduction command and checkpoint limitations.
