"""Geometry review groups are spatial aids, never defect ground truth."""

from infradetect.tbbr_review import group_geometry_candidates


def _candidate(index, x):
    """Make the minimum browser candidate with an original PLY point index."""

    return {"id": index, "xyz": [x, 0.0, 0.0],
            "geometry_score": 1.0, "observations": []}


def test_nearby_candidates_group_without_joining_distant_points():
    """Nearby points form a group; the 1%-of-cloud cap prevents a long bridge."""

    candidates = [_candidate(10, 0), _candidate(11, 0.005),
                  _candidate(50, 0.5), _candidate(51, 0.505)]
    groups = group_geometry_candidates(candidates, diagonal=1.0)
    assert {tuple(group["point_ids"]) for group in groups} == {(10, 11), (50, 51)}
    assert candidates[0]["group_id"] == candidates[1]["group_id"]
    assert candidates[0]["group_id"] != candidates[2]["group_id"]


def test_single_geometry_candidate_remains_reviewable():
    """An isolated point remains visible as a one-point review group."""

    groups = group_geometry_candidates([_candidate(7, 2)], diagonal=5.0)
    assert groups[0]["point_ids"] == [7]
