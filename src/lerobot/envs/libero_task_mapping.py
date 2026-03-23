"""
LIBERO task text to dataset task_index mapping.

This mapping is extracted from the LIBERO dataset at:
/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero/meta/tasks.parquet

The dataset contains 40 tasks (task_index 0-39), and the task descriptions are NOT in sequential order.
Environment task_ids (per-suite 0-9) do not directly map to dataset task_index.

During evaluation, LIBERO environments provide per-suite task_ids (0-9 within each suite),
but the CARP model expects dataset task_index (0-39) that it saw during training.
This module provides the mapping between task descriptions and dataset task indices.
"""

# Mapping from task description (env.task_description) to dataset task_index
LIBERO_TASK_TEXT_TO_INDEX = {
    "put the white mug on the left plate and put the yellow and white mug on the right plate": 0,
    "put the white mug on the plate and put the chocolate pudding to the right of the plate": 1,
    "put the yellow and white mug in the microwave and close it": 2,
    "turn on the stove and put the moka pot on it": 3,
    "put both the alphabet soup and the cream cheese box in the basket": 4,
    "put both the alphabet soup and the tomato sauce in the basket": 5,
    "put both moka pots on the stove": 6,
    "put both the cream cheese box and the butter in the basket": 7,
    "put the black bowl in the bottom drawer of the cabinet and close it": 8,
    "pick up the book and place it in the back compartment of the caddy": 9,
    "put the bowl on the plate": 10,
    "put the wine bottle on the rack": 11,
    "open the top drawer and put the bowl inside": 12,
    "put the cream cheese in the bowl": 13,
    "put the wine bottle on top of the cabinet": 14,
    "push the plate to the front of the stove": 15,
    "turn on the stove": 16,
    "put the bowl on the stove": 17,
    "put the bowl on top of the cabinet": 18,
    "open the middle drawer of the cabinet": 19,
    "pick up the orange juice and place it in the basket": 20,
    "pick up the ketchup and place it in the basket": 21,
    "pick up the cream cheese and place it in the basket": 22,
    "pick up the bbq sauce and place it in the basket": 23,
    "pick up the alphabet soup and place it in the basket": 24,
    "pick up the milk and place it in the basket": 25,
    "pick up the salad dressing and place it in the basket": 26,
    "pick up the butter and place it in the basket": 27,
    "pick up the tomato sauce and place it in the basket": 28,
    "pick up the chocolate pudding and place it in the basket": 29,
    "pick up the black bowl next to the cookie box and place it on the plate": 30,
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate": 31,
    "pick up the black bowl on the ramekin and place it on the plate": 32,
    "pick up the black bowl on the stove and place it on the plate": 33,
    "pick up the black bowl between the plate and the ramekin and place it on the plate": 34,
    "pick up the black bowl on the cookie box and place it on the plate": 35,
    "pick up the black bowl next to the plate and place it on the plate": 36,
    "pick up the black bowl next to the ramekin and place it on the plate": 37,
    "pick up the black bowl from table center and place it on the plate": 38,
    "pick up the black bowl on the wooden cabinet and place it on the plate": 39,
}


def get_task_index_from_description(task_description: str) -> int | None:
    """
    Get dataset task_index from LIBERO task description.

    Args:
        task_description: Task description string from env.task_description

    Returns:
        task_index (0-39) if found, None otherwise

    Example:
        >>> desc = "pick up the alphabet soup and place it in the basket"
        >>> get_task_index_from_description(desc)
        24
    """
    return LIBERO_TASK_TEXT_TO_INDEX.get(task_description)
