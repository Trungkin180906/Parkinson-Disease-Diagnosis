def risk_level(score: float) -> str:
    # Continuous intervals also cover fractional scores without gaps.
    if score <= 30:
        return "low"
    if score <= 60:
        return "medium"
    if score <= 80:
        return "high"
    return "very_high"


def recommendation(score, threshold):
    if score >= threshold:
        return "Phát hiện dấu hiệu nguy cơ cao; khuyến nghị đến cơ sở y tế để bác sĩ đánh giá."
    return "Chưa ghi nhận nhiều dấu hiệu nguy cơ qua lần sàng lọc này; tiếp tục theo dõi định kỳ."


def voice_score(updrs, reference_max):
    # Project display index, not a clinical scale or a probability.
    return round(max(0, min(100, 100 * (1 - updrs / reference_max))), 2)


def voice_history(rows):
    result = []
    previous = None
    for source in rows:
        item = dict(source)
        item.update(delta_updrs=None, delta_voice_score=None, trend="baseline", comparison_record_id=None)
        if previous is not None:
            if (item["model_version"], item["feature_schema"], item["reference_max"]) != (
                previous["model_version"],
                previous["feature_schema"],
                previous["reference_max"],
            ):
                item["trend"] = "not_comparable"
            else:
                delta = round(item["total_updrs"] - previous["total_updrs"], 4)
                threshold = item["change_threshold"]
                item.update(
                    delta_updrs=delta,
                    delta_voice_score=round(item["voice_score"] - previous["voice_score"], 2),
                    trend="increased"
                    if delta >= threshold
                    else "decreased"
                    if delta <= -threshold
                    else "stable",
                    comparison_record_id=previous["id"],
                )
        result.append(item)
        previous = item
    return result
