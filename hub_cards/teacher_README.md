---
base_model: Qwen/Qwen2-VL-7B-Instruct
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

# Vietnamese VQA Teacher 7B

LoRA adapter maintained by **Ouyangxuelili** for Vietnamese visual question
answering. Load it on top of `Qwen/Qwen2-VL-7B-Instruct` with PEFT.

This repository contains adapter and processor files, not a merged copy of the
base model.

## Intended use

Vietnamese image question answering and comparison against the lightweight 2B
student adapter.

## Base model

Qwen/Qwen2-VL-7B-Instruct (Apache-2.0).

## Maintainer

Ouyangxuelili
