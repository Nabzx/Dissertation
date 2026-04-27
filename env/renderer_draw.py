from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def _update_env(
    self,
    grid: np.ndarray,
    actions: Optional[Dict[str, int]] = None,
    communication_events: Optional[List[Dict[str, object]]] = None,
    render_info: Optional[Dict[str, object]] = None,
) -> None:
    self._update_facing(actions)
    self._update_perception(grid)
    self._update_resource_animation_state(grid)
    draw_obstacles(self, grid)
    draw_resources(self, grid)
    now_positions = draw_agents(self, grid)
    draw_pulses(self, now_positions, communication_events)
    self._update_game_mode_visuals(render_info, now_positions)


def draw_obstacles(self, grid: np.ndarray) -> None:
    obstacle_mask = grid == self.obstacle_value
    for patch, is_obstacle in zip(self.obstacle_patches, obstacle_mask.flatten()):
        row = int(patch.get_y() - 0.12)
        col = int(patch.get_x() - 0.12)
        patch.set_visible(bool(is_obstacle) and self.arena_mask[row, col])


def draw_resources(self, grid: np.ndarray) -> None:
    resource_idx = 0
    for row, col in np.argwhere(grid == 1):
        if resource_idx >= len(self.resource_patches):
            break
        if not self.arena_mask[row, col]:
            continue
        patch = self.resource_patches[resource_idx]
        patch.center = (float(col) + 0.5, float(row) + 0.5)
        spawn_ttl = self.resource_spawn_state.get((row, col))
        if self.show_resource_animation and spawn_ttl is not None:
            progress = 1.0 - ((spawn_ttl - 1) / float(self.resource_anim_steps))
            patch.set_radius(0.08 + 0.10 * progress)
            patch.set_alpha(0.45 + 0.55 * progress)
        else:
            patch.set_radius(0.18)
            patch.set_alpha(1.0)
        patch.set_visible(True)
        resource_idx += 1

    for idx in range(resource_idx, len(self.resource_patches)):
        self.resource_patches[idx].set_visible(False)

    self._apply_resource_animations()


def draw_agents(self, grid: np.ndarray) -> Dict[int, tuple[int, int]]:
    now_positions: Dict[int, tuple[int, int]] = {}
    for agent_value, patch in self.agent_patches.items():
        positions = np.argwhere(grid == agent_value)
        if len(positions) == 0:
            patch.set_visible(False)
            self.agent_trails[agent_value].clear()
            self._update_trail_patches(agent_value)
            continue
        row, col = positions[0]
        if not self.arena_mask[row, col]:
            patch.set_visible(False)
            self.agent_trails[agent_value].clear()
            self._update_trail_patches(agent_value)
            continue
        self._append_trail_position(agent_value, int(row), int(col))
        patch.center = (float(col) + 0.5, float(row) + 0.5)
        patch.set_visible(True)
        now_positions[agent_value] = (int(row), int(col))
    return now_positions


def _update_game_mode_visuals(
    self,
    render_info: Optional[Dict[str, object]],
    now_positions: Dict[int, tuple[int, int]],
) -> None:
    self.flag_patch.set_visible(False)
    self.flag_glow_patch.set_visible(False)
    self.winner_highlight_patch.set_visible(False)

    if not render_info:
        return

    flag_position = render_info.get("flag_position")
    if flag_position is not None:
        row, col = int(flag_position[0]), int(flag_position[1])
        if 0 <= row < self.arena_mask.shape[0] and 0 <= col < self.arena_mask.shape[1]:
            if self.arena_mask[row, col]:
                cx, cy = float(col) + 0.5, float(row) + 0.5
                self.flag_patch.set_xy([
                    (cx - 0.20, cy + 0.24),
                    (cx + 0.26, cy + 0.08),
                    (cx - 0.20, cy - 0.08),
                ])
                self.flag_patch.set_visible(True)
                self.flag_glow_patch.center = (cx, cy)
                self.flag_glow_patch.set_alpha(0.22)
                self.flag_glow_patch.set_visible(True)

    winner = render_info.get("winner")
    if winner:
        winner_value = self._agent_value(str(winner))
        winner_position = now_positions.get(winner_value)
        if winner_position is not None:
            row, col = winner_position
            self.winner_highlight_patch.center = (float(col) + 0.5, float(row) + 0.5)
            self.winner_highlight_patch.set_visible(True)

def _update_resource_animation_state(self, grid: np.ndarray) -> None:
    if not self.show_resource_animation:
        self.previous_resource_positions = set(map(tuple, np.argwhere(grid == 1)))
        self.resource_spawn_state.clear()
        self.resource_collect_state.clear()
        return

    current_resources = {tuple(pos) for pos in np.argwhere(grid == 1)}
    spawned = current_resources - self.previous_resource_positions
    collected = self.previous_resource_positions - current_resources

    for pos in spawned:
        self.resource_spawn_state[pos] = self.resource_anim_steps
    for pos in collected:
        self.resource_collect_state[pos] = self.resource_anim_steps

    self.previous_resource_positions = current_resources

def _apply_resource_animations(self) -> None:
    for patch in self.resource_glow_patches:
        patch.set_visible(False)
    for patch in self.resource_collect_patches:
        patch.set_visible(False)

    if not self.show_resource_animation:
        return

    active_spawns = list(self.resource_spawn_state.items())[: len(self.resource_glow_patches)]
    for idx, (pos, ttl) in enumerate(active_spawns):
        row, col = pos
        if not self.arena_mask[row, col]:
            continue
        progress = 1.0 - ((ttl - 1) / float(self.resource_anim_steps))
        glow_patch = self.resource_glow_patches[idx]
        glow_patch.center = (float(col) + 0.5, float(row) + 0.5)
        glow_patch.set_radius(0.18 + 0.22 * progress)
        glow_patch.set_alpha(0.35 * (1.0 - progress / 1.2))
        glow_patch.set_visible(True)

    active_collects = list(self.resource_collect_state.items())[: len(self.resource_collect_patches)]
    for idx, (pos, ttl) in enumerate(active_collects):
        row, col = pos
        if not self.arena_mask[row, col]:
            continue
        progress = 1.0 - ((ttl - 1) / float(self.resource_anim_steps))
        collect_patch = self.resource_collect_patches[idx]
        collect_patch.center = (float(col) + 0.5, float(row) + 0.5)
        collect_patch.set_radius(0.16 + 0.18 * progress)
        collect_patch.set_alpha(0.42 * (1.0 - progress))
        collect_patch.set_visible(True)

    self.resource_spawn_state = {
        pos: ttl - 1 for pos, ttl in self.resource_spawn_state.items() if ttl > 1
    }
    self.resource_collect_state = {
        pos: ttl - 1 for pos, ttl in self.resource_collect_state.items() if ttl > 1
    }

def _append_trail_position(self, agent_value: int, row: int, col: int) -> None:
    trail = self.agent_trails[agent_value]
    new_position = (row, col)
    if not trail or trail[-1] != new_position:
        trail.append(new_position)
    self._update_trail_patches(agent_value)

def _update_trail_patches(self, agent_value: int) -> None:
    trail = list(self.agent_trails[agent_value])
    patches = self.trail_patches[agent_value]
    visible_trail = trail[:-1] if len(trail) > 1 else []

    for patch in patches:
        patch.set_visible(False)

    if not visible_trail:
        return

    for idx, (row, col) in enumerate(reversed(visible_trail)):
        if idx >= len(patches):
            break
        if not self.arena_mask[row, col]:
            continue
        fade = max(0.12, 0.55 * (1.0 - idx / max(len(visible_trail), 1)))
        patch = patches[idx]
        patch.center = (float(col) + 0.5, float(row) + 0.5)
        patch.set_alpha(fade)
        patch.set_visible(True)

def _update_facing(self, actions: Optional[Dict[str, int]]) -> None:
    if not actions:
        return
    action_to_direction = {
        1: (-1, 0),
        2: (1, 0),
        3: (0, -1),
        4: (0, 1),
    }
    for agent_id, action in actions.items():
        if action not in action_to_direction:
            continue
        agent_value = self._agent_value(agent_id)
        if agent_value not in self.agent_facing:
            continue
        self.agent_facing[agent_value] = action_to_direction[action]

def _update_perception(self, grid: np.ndarray) -> None:
    if not self.show_perception:
        return

    for agent_value in self.perception_ray_lines.keys():
        for patch in self.perception_cell_patches[agent_value]:
            patch.set_visible(False)
        for line in self.perception_ray_lines[agent_value]:
            line.set_visible(False)

    for agent_value in self.perception_ray_lines.keys():
        positions = np.argwhere(grid == agent_value)
        if len(positions) == 0:
            continue

        row, col = positions[0]
        facing = self.agent_facing[agent_value]
        ray_specs = self._compute_forward_rays(int(row), int(col), facing, grid)

        visible_cells = []
        for idx, ray in enumerate(ray_specs):
            if idx < len(self.perception_ray_lines[agent_value]) and len(ray["points"]) >= 2:
                xs = [point[0] for point in ray["points"]]
                ys = [point[1] for point in ray["points"]]
                line = self.perception_ray_lines[agent_value][idx]
                line.set_data(xs, ys)
                line.set_visible(True)
            visible_cells.extend(ray["cells"])

        unique_cells = []
        seen = set()
        for cell in visible_cells:
            if cell in seen:
                continue
            seen.add(cell)
            unique_cells.append(cell)

        for idx, (cell_row, cell_col) in enumerate(unique_cells[: len(self.perception_cell_patches[agent_value])]):
            patch = self.perception_cell_patches[agent_value][idx]
            patch.set_xy((cell_col, cell_row))
            patch.set_width(1)
            patch.set_height(1)
            patch.set_visible(True)

def _compute_forward_rays(
    self,
    row: int,
    col: int,
    facing: tuple[int, int],
    grid: np.ndarray,
) -> List[Dict[str, List]]:
    dr, dc = facing
    if (dr, dc) in [(0, 1), (0, -1)]:
        fan_offsets = [-0.45, 0.0, 0.45]
        lateral = (1, 0)
    else:
        fan_offsets = [-0.45, 0.0, 0.45]
        lateral = (0, 1)

    origin_x = float(col) + 0.5
    origin_y = float(row) + 0.5
    rays = []

    for offset in fan_offsets:
        points = [(origin_x, origin_y)]
        cells = []
        blocked = False
        for step in range(1, self.perception_range + 1):
            target_row = row + dr * step + int(round(lateral[0] * offset * step))
            target_col = col + dc * step + int(round(lateral[1] * offset * step))

            if not (0 <= target_row < self.grid_rows and 0 <= target_col < self.grid_cols):
                break
            if not self.arena_mask[target_row, target_col]:
                break

            cell_centre = (float(target_col) + 0.5, float(target_row) + 0.5)
            points.append(cell_centre)
            cells.append((target_row, target_col))

            if grid[target_row, target_col] == self.obstacle_value:
                blocked = True
                break

        if blocked or len(points) > 1:
            rays.append({"points": points, "cells": cells})

    return rays

def _update_communication_visuals(
    self,
    positions: Dict[int, tuple[int, int]],
    communication_events: Optional[List[Dict[str, object]]],
) -> None:
    if not self.show_communication:
        self.pulses = []
        for patch in self.communication_pulse_patches:
            patch.set_visible(False)
        return

    if communication_events:
        for event in communication_events:
            sender = int(event["sender"])
            if sender in positions:
                row, col = positions[sender]
                pos = (float(col) + 0.5, float(row) + 0.5)
                self.pulses.append({"pos": pos, "r": self.communication_start_radius, "a": 0.6})

    live_pulses = []
    for pulse in self.pulses:
        pulse["r"] = float(pulse["r"]) + self.communication_growth_rate
        pulse["a"] = float(pulse["a"]) - self.communication_fade_rate
        if pulse["a"] > 0:
            live_pulses.append(pulse)
    self.pulses = live_pulses[-len(self.communication_pulse_patches):]

    for patch, pulse in zip(self.communication_pulse_patches, self.pulses):
        patch.center = pulse["pos"]
        patch.set_radius(float(pulse["r"]))
        patch.set_alpha(float(pulse["a"]))
        patch.set_visible(True)
    for patch in self.communication_pulse_patches[len(self.pulses):]:
        patch.set_visible(False)

def reset_communication_visuals(self) -> None:
    self.pulses = []
    for patch in self.communication_pulse_patches:
        patch.set_visible(False)


def draw_env(self, grid, actions=None, communication_events=None, render_info=None):
    return _update_env(self, grid, actions, communication_events, render_info)


def draw_pulses(self, positions, comms=None):
    return _update_communication_visuals(self, positions, comms)
