/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

pub(crate) const MINIMUM_TRACKING_FOR_DRAG: f64 = 5.0;
pub(crate) const ACCELERATION: f64 = 15.0;
pub(crate) const MAX_TRACKING_TIME_MS: f64 = 100.0;
pub(crate) const DECELERATION_FRICTION_FACTOR: f64 = 0.95;
pub(crate) const DESIRED_FRAME_TIME_MS: f64 = 1000.0 / 60.0;
pub(crate) const MINIMUM_VELOCITY: f64 = 0.01;
pub(crate) const PENETRATION_DECELERATION: f64 = 0.03;
pub(crate) const PENETRATION_ACCELERATION: f64 = 0.08;
pub(crate) const MINIMUM_VELOCITY_FOR_DECELERATION: f64 = 1.0;
pub(crate) const SNAP_DURATION_MS: f64 = 250.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ScrollDeceleration {
    pub offset: f64,
    pub velocity: f64,
    pub min_offset: f64,
    pub decelerating: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ScrollTracking {
    pub offset: f64,
    pub min_offset: f64,
    pub tracking: bool,
    pub dragging: bool,
    pub start_offset: f64,
    pub start_touch_position: f64,
    pub last_touch_position: f64,
    pub start_time: f64,
    pub start_time_offset: f64,
    pub last_move_time: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ScrollMove {
    pub state: ScrollTracking,
    pub began_dragging: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ScrollRelease {
    pub state: ScrollTracking,
    pub deceleration: ScrollDeceleration,
    pub was_dragging: bool,
    pub should_snap: bool,
}

#[cfg(test)]
pub(crate) fn minimum_scroll_offset(viewport_size: f64, content_size: f64) -> f64 {
    viewport_size - viewport_size.max(content_size)
}

pub(crate) fn clamp_scroll_offset(offset: f64, min_offset: f64) -> f64 {
    offset.max(min_offset).min(0.0)
}

pub(crate) fn rubber_band_scroll_offset(offset: f64, min_offset: f64) -> f64 {
    let penetration = if offset < min_offset {
        offset - min_offset
    } else if offset > 0.0 {
        offset
    } else {
        0.0
    };
    offset - penetration / 2.0
}

pub(crate) fn tracked_scroll_offset(start_offset: f64, touch_delta: f64, min_offset: f64) -> f64 {
    rubber_band_scroll_offset(start_offset + touch_delta, min_offset)
}

pub(crate) fn begin_scroll_tracking(
    offset: f64,
    min_offset: f64,
    touch_position: f64,
    time: f64,
) -> ScrollTracking {
    let bounded_offset = clamp_scroll_offset(offset, min_offset);
    ScrollTracking {
        offset: bounded_offset,
        min_offset,
        tracking: true,
        dragging: false,
        start_offset: bounded_offset,
        start_touch_position: touch_position,
        last_touch_position: touch_position,
        start_time: time,
        start_time_offset: bounded_offset,
        last_move_time: None,
    }
}

pub(crate) fn move_scroll_tracking(
    state: ScrollTracking,
    touch_position: f64,
    time: f64,
) -> ScrollMove {
    if !state.tracking {
        return ScrollMove {
            state,
            began_dragging: false,
        };
    }

    let touch_delta = touch_position - state.start_touch_position;
    if !state.dragging && touch_delta.abs() >= MINIMUM_TRACKING_FOR_DRAG {
        return ScrollMove {
            state: ScrollTracking {
                dragging: true,
                start_touch_position: touch_position,
                last_touch_position: touch_position,
                ..state
            },
            began_dragging: true,
        };
    }
    if !state.dragging {
        return ScrollMove {
            state,
            began_dragging: false,
        };
    }

    let offset = tracked_scroll_offset(state.start_offset, touch_delta, state.min_offset);
    let reset_velocity_window = time - state.start_time > MAX_TRACKING_TIME_MS;
    ScrollMove {
        state: ScrollTracking {
            offset,
            last_touch_position: touch_position,
            start_time: if reset_velocity_window {
                time
            } else {
                state.start_time
            },
            start_time_offset: if reset_velocity_window {
                offset
            } else {
                state.start_time_offset
            },
            last_move_time: Some(time),
            ..state
        },
        began_dragging: false,
    }
}

pub(crate) fn scroll_release_velocity(offset: f64, start_offset: f64, elapsed_ms: f64) -> f64 {
    if elapsed_ms <= 0.0 {
        return 0.0;
    }
    (offset - start_offset) / (elapsed_ms / ACCELERATION)
}

pub(crate) fn should_track_scroll_release(last_move_time: f64, release_time: f64) -> bool {
    release_time - last_move_time <= MAX_TRACKING_TIME_MS
}

pub(crate) fn end_scroll_tracking(
    state: ScrollTracking,
    time: f64,
    touch_position: Option<f64>,
) -> ScrollRelease {
    let release_state = match touch_position {
        Some(position) if state.dragging && position != state.last_touch_position => {
            move_scroll_tracking(state, position, time).state
        }
        _ => state,
    };
    let was_dragging = release_state.dragging;
    let stopped = ScrollTracking {
        tracking: false,
        dragging: false,
        ..release_state
    };
    let velocity = match (was_dragging, release_state.last_move_time) {
        (true, Some(last_move_time)) if should_track_scroll_release(last_move_time, time) => {
            scroll_release_velocity(
                release_state.offset,
                release_state.start_time_offset,
                time - release_state.start_time,
            )
        }
        _ => 0.0,
    };
    let deceleration =
        start_scroll_deceleration(release_state.offset, velocity, release_state.min_offset);
    ScrollRelease {
        state: stopped,
        deceleration,
        was_dragging,
        should_snap: !deceleration.decelerating,
    }
}

pub(crate) fn start_scroll_deceleration(
    offset: f64,
    velocity: f64,
    min_offset: f64,
) -> ScrollDeceleration {
    ScrollDeceleration {
        offset,
        velocity,
        min_offset,
        decelerating: velocity.abs() > MINIMUM_VELOCITY_FOR_DECELERATION,
    }
}

pub(crate) fn step_scroll_deceleration(state: ScrollDeceleration) -> ScrollDeceleration {
    if !state.decelerating {
        return state;
    }

    let offset = state.offset + state.velocity;
    let mut velocity = state.velocity * DECELERATION_FRICTION_FACTOR;
    if velocity.abs() <= MINIMUM_VELOCITY {
        return ScrollDeceleration {
            offset,
            velocity,
            decelerating: false,
            ..state
        };
    }

    let penetration = if offset < state.min_offset {
        state.min_offset - offset
    } else if offset > 0.0 {
        -offset
    } else {
        0.0
    };
    if penetration != 0.0 {
        if penetration * velocity <= 0.0 {
            velocity += penetration * PENETRATION_DECELERATION;
        } else {
            velocity = penetration * PENETRATION_ACCELERATION;
        }
    }

    ScrollDeceleration {
        offset,
        velocity,
        ..state
    }
}

pub(crate) fn step_scroll_deceleration_frames(
    state: ScrollDeceleration,
    frame_count: u32,
) -> ScrollDeceleration {
    let mut next = state;
    for _ in 0..frame_count {
        next = step_scroll_deceleration(next);
        if !next.decelerating {
            break;
        }
    }
    next
}

pub(crate) fn missed_scroll_frame_count(elapsed_ms: f64) -> u32 {
    ((elapsed_ms / DESIRED_FRAME_TIME_MS).round() - 1.0).max(0.0) as u32
}

pub(crate) fn ease_snap_progress(elapsed_ms: f64) -> f64 {
    let target_x = (elapsed_ms / SNAP_DURATION_MS).clamp(0.0, 1.0);
    if target_x == 0.0 || target_x == 1.0 {
        return target_x;
    }

    let sample = |time: f64, first: f64, second: f64| {
        let inverse = 1.0 - time;
        3.0 * inverse * inverse * time * first
            + 3.0 * inverse * time * time * second
            + time * time * time
    };
    let mut low = 0.0;
    let mut high = 1.0;
    for _ in 0..24 {
        let middle = (low + high) / 2.0;
        if sample(middle, 0.25, 0.25) < target_x {
            low = middle;
        } else {
            high = middle;
        }
    }
    sample((low + high) / 2.0, 0.1, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ios_scroll_constants_match_the_pastrykit_model() {
        assert_eq!(MINIMUM_TRACKING_FOR_DRAG, 5.0);
        assert_eq!(ACCELERATION, 15.0);
        assert_eq!(MAX_TRACKING_TIME_MS, 100.0);
        assert_eq!(DECELERATION_FRICTION_FACTOR, 0.95);
        assert_eq!(DESIRED_FRAME_TIME_MS, 1000.0 / 60.0);
        assert_eq!(MINIMUM_VELOCITY, 0.01);
        assert_eq!(PENETRATION_DECELERATION, 0.03);
        assert_eq!(PENETRATION_ACCELERATION, 0.08);
        assert_eq!(MINIMUM_VELOCITY_FOR_DECELERATION, 1.0);
        assert_eq!(SNAP_DURATION_MS, 250.0);
    }

    #[test]
    fn minimum_scroll_offset_keeps_a_fitted_list_at_zero() {
        assert_eq!(minimum_scroll_offset(500.0, 800.0), -300.0);
        assert_eq!(minimum_scroll_offset(500.0, 300.0), 0.0);
    }

    #[test]
    fn rubber_band_halves_penetration_on_both_edges() {
        assert_eq!(rubber_band_scroll_offset(20.0, -300.0), 10.0);
        assert_eq!(rubber_band_scroll_offset(-340.0, -300.0), -320.0);
        assert_eq!(rubber_band_scroll_offset(-100.0, -300.0), -100.0);
    }

    #[test]
    fn a_fitted_list_still_tracks_a_rubber_band_drag() {
        assert_eq!(tracked_scroll_offset(0.0, 40.0, 0.0), 20.0);
        assert_eq!(tracked_scroll_offset(0.0, -40.0, 0.0), -20.0);
    }

    #[test]
    fn tracking_clamps_the_resting_offset_before_a_new_gesture() {
        assert_eq!(
            begin_scroll_tracking(20.0, -300.0, 100.0, 0.0),
            ScrollTracking {
                offset: 0.0,
                min_offset: -300.0,
                tracking: true,
                dragging: false,
                start_offset: 0.0,
                start_touch_position: 100.0,
                last_touch_position: 100.0,
                start_time: 0.0,
                start_time_offset: 0.0,
                last_move_time: None,
            }
        );
    }

    #[test]
    fn tracking_uses_the_exact_drag_threshold_and_swallows_its_first_move() {
        let start = begin_scroll_tracking(0.0, 0.0, 100.0, 0.0);
        let below_threshold = move_scroll_tracking(start, 96.0, 10.0);
        assert_eq!(below_threshold.state, start);
        assert!(!below_threshold.began_dragging);

        let began = move_scroll_tracking(start, 95.0, 20.0);
        assert_eq!(began.state.offset, 0.0);
        assert_eq!(began.state.start_touch_position, 95.0);
        assert!(began.began_dragging);

        let dragged = move_scroll_tracking(began.state, 75.0, 30.0);
        assert_eq!(dragged.state.offset, -10.0);
    }

    #[test]
    fn tracking_resets_the_exact_release_velocity_window() {
        let start = begin_scroll_tracking(0.0, 0.0, 100.0, 0.0);
        let began = move_scroll_tracking(start, 95.0, 10.0).state;
        let first = move_scroll_tracking(began, 75.0, 20.0).state;
        let reset = move_scroll_tracking(first, 55.0, 120.0).state;
        let latest = move_scroll_tracking(reset, 35.0, 130.0).state;
        let release = end_scroll_tracking(latest, 140.0, None);

        assert_eq!(reset.start_time, 120.0);
        assert_eq!(reset.start_time_offset, -20.0);
        assert_eq!(release.deceleration.velocity, -7.5);
        assert!(release.deceleration.decelerating);
        assert!(!release.should_snap);
    }

    #[test]
    fn tracking_springs_instead_of_flinging_after_a_stale_release() {
        let start = begin_scroll_tracking(0.0, 0.0, 100.0, 0.0);
        let began = move_scroll_tracking(start, 95.0, 10.0).state;
        let dragged = move_scroll_tracking(began, 75.0, 20.0).state;
        let release = end_scroll_tracking(dragged, 121.0, None);

        assert!(release.was_dragging);
        assert_eq!(release.deceleration.velocity, 0.0);
        assert!(!release.deceleration.decelerating);
        assert!(release.should_snap);
    }

    #[test]
    fn tracking_consumes_a_coalesced_final_touch() {
        let start = begin_scroll_tracking(0.0, -500.0, 700.0, 0.0);
        let delivered_move = move_scroll_tracking(start, 649.0, 20.0).state;
        let release = end_scroll_tracking(delivered_move, 40.0, Some(610.0));

        assert_eq!(release.state.offset, -39.0);
        assert_eq!(release.deceleration.velocity, -14.625);
        assert!(release.deceleration.decelerating);
    }

    #[test]
    fn tracking_does_not_reprocess_an_unchanged_final_touch() {
        let start = begin_scroll_tracking(0.0, 0.0, 300.0, 0.0);
        let began = move_scroll_tracking(start, 350.0, 50.0).state;
        let moved = move_scroll_tracking(began, 400.0, 90.0).state;
        let release = end_scroll_tracking(moved, 110.0, Some(400.0));

        assert_eq!(release.state.offset, 25.0);
        assert_eq!(release.deceleration.velocity, 25.0 / (110.0 / 15.0));
        assert!(release.deceleration.decelerating);
    }

    #[test]
    fn scroll_release_uses_the_100ms_velocity_window() {
        assert_eq!(scroll_release_velocity(-45.0, 0.0, 100.0), -6.75);
        assert!(should_track_scroll_release(100.0, 200.0));
        assert!(!should_track_scroll_release(100.0, 201.0));
    }

    #[test]
    fn deceleration_moves_then_applies_friction() {
        assert_eq!(
            step_scroll_deceleration(start_scroll_deceleration(-100.0, -10.0, -300.0)),
            ScrollDeceleration {
                offset: -110.0,
                velocity: -9.5,
                min_offset: -300.0,
                decelerating: true,
            }
        );
    }

    #[test]
    fn deceleration_applies_the_exact_penetration_spring() {
        assert_eq!(
            step_scroll_deceleration(start_scroll_deceleration(5.0, 2.0, -300.0)),
            ScrollDeceleration {
                offset: 7.0,
                velocity: 1.69,
                min_offset: -300.0,
                decelerating: true,
            }
        );
        assert_eq!(
            step_scroll_deceleration(start_scroll_deceleration(5.0, -2.0, -300.0)),
            ScrollDeceleration {
                offset: 3.0,
                velocity: -0.24,
                min_offset: -300.0,
                decelerating: true,
            }
        );
    }

    #[test]
    fn deceleration_stops_below_the_exact_minimum_velocity() {
        let state = step_scroll_deceleration(start_scroll_deceleration(-100.0, 2.0, -300.0));
        let stopped = step_scroll_deceleration_frames(state, 200);
        assert!(!stopped.decelerating);
        assert!(stopped.velocity.abs() <= MINIMUM_VELOCITY);
    }

    #[test]
    fn missed_animation_frames_use_the_pastrykit_round_rule() {
        assert_eq!(missed_scroll_frame_count(16.0), 0);
        assert_eq!(missed_scroll_frame_count(34.0), 1);
        assert_eq!(missed_scroll_frame_count(50.0), 2);
    }

    #[test]
    fn clamp_scroll_offset_returns_the_nearest_edge() {
        assert_eq!(clamp_scroll_offset(20.0, -300.0), 0.0);
        assert_eq!(clamp_scroll_offset(-340.0, -300.0), -300.0);
        assert_eq!(clamp_scroll_offset(-100.0, -300.0), -100.0);
    }

    #[test]
    fn ease_snap_matches_css_ease_endpoints_and_is_monotonic() {
        assert_eq!(ease_snap_progress(0.0), 0.0);
        assert_eq!(ease_snap_progress(SNAP_DURATION_MS), 1.0);
        let samples = [50.0, 100.0, 150.0, 200.0].map(ease_snap_progress);
        assert!(samples.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn touch_handler_uses_the_native_scroll_model() {
        assert!(include_str!("touch.rs").contains("begin_scroll_tracking"));
        assert!(!include_str!("touch.rs").contains("poor-mans acceleration"));
    }
}
