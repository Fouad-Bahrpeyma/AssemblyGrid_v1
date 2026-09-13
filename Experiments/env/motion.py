"""Canonical AssemblyGrid v1 abstract geometric motion proxy.

The module is deliberately dependency-free. Every fixed-base robot must reach
a common operation/transfer target; an internal six-coordinate proxy supports
deterministic timing/state evolution; travel duration depends on nominal
geometric motion and speed; and simultaneous straight-line approach segments
are checked with a declared clearance rule. These quantities are benchmark
geometry only: they are not URDF/IK, link-level collision, dynamics, contact,
or physical robot validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, ceil, cos, hypot, pi, sin
from time import perf_counter
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

Point = Tuple[float, float]
Segment = Tuple[Point, Point]
Coord = Tuple[int, int]


@dataclass(frozen=True)
class RobotMotion:
    robot: Coord
    start: Point
    target: Point
    q_target: Tuple[float, ...]
    travel_distance: float
    travel_duration: int
    reach_margin: float
    segment: Segment


@dataclass(frozen=True)
class MotionPlan:
    target: Point
    robot_motions: Tuple[RobotMotion, ...]
    approach_duration: int
    process_duration: int
    retreat_duration: int
    total_duration: int
    min_reach_margin: float
    planning_seconds: float

    @property
    def segments(self) -> Tuple[Segment, ...]:
        return tuple(x.segment for x in self.robot_motions)

    @property
    def path_length(self) -> float:
        return sum(x.travel_distance for x in self.robot_motions)


class MotionInfeasible(RuntimeError):
    pass


def base_point(coord: Coord, spacing: float) -> Point:
    i, j = coord
    return (j * spacing, i * spacing)


def common_target(coords: Sequence[Coord], spacing: float) -> Point:
    if not coords:
        raise ValueError("cannot compute target for empty coalition")
    pts = [base_point(c, spacing) for c in coords]
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def role_targets(center: Point, roles: Mapping[Coord, str], offset: float,
                 bases: Optional[Mapping[Coord, Point]] = None) -> Dict[Coord, Point]:
    """Assign distinct deterministic approach points around an operation.

    The benchmark remains task-level and dimensionless, but no longer implies
    that several physical end effectors occupy one identical point. Semantic
    role and coordinate determine ordering, not mapping insertion order.
    """
    ordered = sorted(roles, key=lambda rc: (roles[rc], rc))
    if len(ordered) <= 1 or offset <= 0:
        return {rc: center for rc in ordered}
    phase = -pi / 2
    targets: Dict[Coord, Point] = {}
    for index, rc in enumerate(ordered):
        if bases is not None:
            base = bases[rc]
            dx, dy = base[0] - center[0], base[1] - center[1]
            norm = hypot(dx, dy)
        else:
            norm = 0.0
        if norm > 1e-12:
            # Approach from the robot's own side of the operation. This makes
            # targets distinct without increasing the centre reach required.
            targets[rc] = (center[0] + offset * dx / norm, center[1] + offset * dy / norm)
        else:
            angle = phase + 2 * pi * index / len(ordered)
            targets[rc] = (center[0] + offset * cos(angle), center[1] + offset * sin(angle))
    return targets


def q_proxy(base: Point, target: Point, reach_radius: float) -> Tuple[float, ...]:
    dx, dy = target[0] - base[0], target[1] - base[1]
    angle = atan2(dy, dx) if abs(dx) + abs(dy) > 1e-12 else 0.0
    radius = hypot(dx, dy)
    # Six bounded abstract joint coordinates.  They are intentionally a proxy,
    # but unlike the previous constant q they evolve with task geometry.
    return (
        max(-pi, min(pi, angle)),
        max(-pi, min(pi, (radius / max(reach_radius, 1e-9)) * (pi / 2))),
        -0.35 * max(-pi, min(pi, angle)),
        0.25 * max(-pi, min(pi, angle)),
        0.0,
        0.0,
    )


def plan_motion(robots: Mapping[Coord, object], coords: Sequence[Coord], target: Point,
                spacing: float, process_duration: int, retreat_fraction: float = 0.25,
                targets_by_coord: Optional[Mapping[Coord, Point]] = None) -> MotionPlan:
    tic = perf_counter()
    motions = []
    for coord in coords:
        r = robots[coord]
        base = base_point(coord, spacing)
        robot_target = targets_by_coord.get(coord, target) if targets_by_coord else target
        reach = hypot(robot_target[0] - base[0], robot_target[1] - base[1])
        if reach > r.model.reach_radius + 1e-9:
            raise MotionInfeasible(f"{coord} cannot reach target: {reach:.3f}>{r.model.reach_radius:.3f}")
        start = getattr(r, "ee_pos", base)
        travel = hypot(robot_target[0] - start[0], robot_target[1] - start[1])
        speed = max(1e-6, float(r.model.speed))
        cartesian_ticks = int(ceil(travel / speed))
        q_tgt = q_proxy(base, robot_target, r.model.reach_radius)
        q_cur = tuple(getattr(r, "q", (0.0,) * 6))
        max_joint_delta = max((abs(a - b) for a, b in zip(q_tgt, q_cur)), default=0.0)
        joint_speed = max(1e-6, float(getattr(r.model, "joint_speed", 1.0)))
        joint_ticks = int(ceil(max_joint_delta / joint_speed))
        travel_ticks = max(cartesian_ticks, joint_ticks)
        motions.append(RobotMotion(coord, start, robot_target, q_tgt, travel, travel_ticks,
                                   r.model.reach_radius - reach, (start, robot_target)))
    approach = max((m.travel_duration for m in motions), default=0)
    retreat = int(ceil(approach * max(0.0, retreat_fraction)))
    return MotionPlan(
        target=target,
        robot_motions=tuple(motions),
        approach_duration=approach,
        process_duration=max(1, int(process_duration)),
        retreat_duration=retreat,
        total_duration=max(1, approach + int(process_duration) + retreat),
        min_reach_margin=min((m.reach_margin for m in motions), default=float("inf")),
        planning_seconds=perf_counter() - tic,
    )


def _orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, p: Point) -> bool:
    return min(a[0], b[0]) - 1e-12 <= p[0] <= max(a[0], b[0]) + 1e-12 and min(a[1], b[1]) - 1e-12 <= p[1] <= max(a[1], b[1]) + 1e-12


def segments_intersect(s1: Segment, s2: Segment) -> bool:
    a, b = s1; c, d = s2
    o1, o2, o3, o4 = _orientation(a, b, c), _orientation(a, b, d), _orientation(c, d, a), _orientation(c, d, b)
    if (o1 > 0 > o2 or o2 > 0 > o1) and (o3 > 0 > o4 or o4 > 0 > o3):
        return True
    for o, x, y, p in ((o1, a, b, c), (o2, a, b, d), (o3, c, d, a), (o4, c, d, b)):
        if abs(o) <= 1e-12 and _on_segment(x, y, p):
            return True
    return False


def point_segment_distance(p: Point, seg: Segment) -> float:
    a, b = seg
    vx, vy = b[0] - a[0], b[1] - a[1]
    wx, wy = p[0] - a[0], p[1] - a[1]
    denom = vx * vx + vy * vy
    if denom <= 1e-15:
        return hypot(p[0] - a[0], p[1] - a[1])
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
    q = (a[0] + t * vx, a[1] + t * vy)
    return hypot(p[0] - q[0], p[1] - q[1])


def segment_distance(s1: Segment, s2: Segment) -> float:
    if segments_intersect(s1, s2):
        return 0.0
    return min(
        point_segment_distance(s1[0], s2), point_segment_distance(s1[1], s2),
        point_segment_distance(s2[0], s1), point_segment_distance(s2[1], s1),
    )


def plans_conflict(plan_a: MotionPlan, plan_b: MotionPlan, clearance: float) -> bool:
    for sa in plan_a.segments:
        for sb in plan_b.segments:
            if segment_distance(sa, sb) < clearance - 1e-12:
                return True
    return hypot(plan_a.target[0] - plan_b.target[0], plan_a.target[1] - plan_b.target[1]) < clearance - 1e-12
