def to_displayable_percentage(percentage_val: float) -> str:
    return f'{percentage_val:.2f}%'


def to_displayable_percentage_distribution(
    percentage_distribution: dict[str, float],
) -> dict[str, str]:
    keys_sorted_by_values_percentage_desc = sorted(
        percentage_distribution.keys(),
        key=lambda key: percentage_distribution[key],
        reverse=True,
    )
    return {
        key: to_displayable_percentage(percentage_distribution[key])
        for key in keys_sorted_by_values_percentage_desc
    }
