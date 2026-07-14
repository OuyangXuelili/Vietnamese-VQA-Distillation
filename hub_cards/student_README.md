---
base_model: Qwen/Qwen2-VL-2B-Instruct
library_name: peft
pipeline_tag: image-text-to-text
license: apache-2.0
language:
  - vi
tags:
  - qwen2-vl
  - lora
  - visual-question-answering
---

# Vietnamese VQA Student 2B

Lightweight LoRA adapter maintained by **Ouyangxuelili** for Vietnamese visual
question answering. Load it on top of `Qwen/Qwen2-VL-2B-Instruct` with PEFT.

This repository contains adapter and processor files, not a merged copy of the
base model.

## Intended use

Lower-memory Vietnamese image question answering and interactive demos.

## Base model

Qwen/Qwen2-VL-2B-Instruct (Apache-2.0).

## Maintainer

Ouyangxuelili
