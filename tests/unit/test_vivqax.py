from vqa_explanation.data.vivqax import image_manifest, normalize_record


def test_normalize_record_maps_official_fields() -> None:
    raw = {
        "question_id": "1001",
        "image_id": "1",
        "image_name": "COCO_train2014_000000000001.jpg",
        "question": "Đây là gì?",
        "answer": "mèo",
        "explanation": ["con vật có tai và ria"],
    }
    result = normalize_record(raw, "train")
    assert result["explanations"] == ["con vật có tai và ria"]
    assert result["image_path"].endswith(raw["image_name"])


def test_image_manifest_merges_repeated_images_and_tracks_splits() -> None:
    records = [
        {
            "image_name": "COCO_val2014_000000000001.jpg",
            "image_id": "1",
            "split": "validation",
        },
        {
            "image_name": "COCO_val2014_000000000001.jpg",
            "image_id": "1",
            "split": "test",
        },
    ]
    result = image_manifest(records)
    assert len(result) == 1
    assert result[0]["dataset_splits"] == "test,validation"
    assert result[0]["coco_split"] == "val2014"
