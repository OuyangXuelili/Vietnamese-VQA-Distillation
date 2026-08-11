---
version: beta
name: ViVQA Conversation Shell
description: A familiar conversational interface for Vietnamese visual question answering.
colors:
  canvas: "#FFFFFF"
  sidebar: "#F7F7F8"
  hover: "#ECECF1"
  ink: "#202123"
  muted: "#6E6E80"
  line: "#E5E5E5"
  action: "#111827"
typography:
  title:
    fontFamily: Inter, Segoe UI, sans-serif
    fontSize: 15px
    fontWeight: 650
    lineHeight: 1.4
  body:
    fontFamily: Inter, Segoe UI, sans-serif
    fontSize: 15px
    fontWeight: 400
    lineHeight: 1.6
  label:
    fontFamily: Inter, Segoe UI, sans-serif
    fontSize: 12px
    fontWeight: 600
    lineHeight: 1.35
rounded:
  sm: 8px
  md: 10px
  composer: 25px
  full: 9999px
spacing:
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
components:
  sidebar:
    width: 260px
    backgroundColor: "{colors.sidebar}"
  composer:
    backgroundColor: "{colors.canvas}"
    textColor: "{colors.ink}"
    rounded: "{rounded.composer}"
    minHeight: 62px
  attachment:
    width: 44px
    height: 44px
    rounded: "{rounded.full}"
  send:
    width: 44px
    height: 44px
    rounded: "{rounded.full}"
    backgroundColor: "{colors.action}"
---

## Overview

ViVQA is a conversational visual assistant. Its interaction model should be
immediately familiar to anyone who has used a modern AI chat product: choose or
paste an image, ask a question, read the answer, and continue the conversation.
The interface is restrained and neutral so the image and answer stay central.

## Layout

Desktop uses a persistent 260px sidebar and one centered chat column. The sidebar
contains the new-chat action and question history only. The main area contains a
slim model bar, a scrollable conversation, and a composer anchored to the bottom.
On narrow screens the sidebar is hidden so the chat and composer retain space.

## Composer behavior

The composer is one cohesive surface. A circular plus control opens the image
picker; an image pasted with Ctrl+V is accepted by the same component. Once an
image is present, its thumbnail replaces the plus icon without creating a second
upload panel. The question field expands up to four lines. Response style remains
compact inside the composer and the send action is a single dark circular button.

## Conversation history

Each submitted question becomes a short entry under “Hôm nay”. Starting a new
conversation clears the active image and messages but preserves these titles for
the current browser session. History is supporting navigation and must never be
used as a permanent settings panel.

## Visual language

Use white and cool neutral surfaces, one dark action color, thin borders, and one
subtle composer shadow. Assistant messages remain borderless. User messages use a
quiet gray bubble. Avoid gradients, glass effects, colorful cards, oversized
branding, decorative status badges, and research-dashboard terminology.

## Do's and Don'ts

- Do keep the composer visible at common laptop heights.
- Do keep attachment, question, response style, and send action in one row.
- Do show the selected image as a compact thumbnail inside the composer.
- Do preserve question titles when a new chat is created.
- Don't expose token controls in the primary interface.
- Don't add suggestion cards or a dedicated upload canvas.
- Don't use the sidebar for model or generation settings.
