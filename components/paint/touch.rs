/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::time::Instant;

use embedder_traits::{InputEventId, PaintHitTestResult, TouchEventType, TouchId};
use euclid::{Point2D, Scale, Vector2D};
use log::{debug, error, warn};
use paint_api::display_list::TouchAction;
use rustc_hash::{FxHashMap, FxHashSet};
use servo_base::id::{PipelineId, WebViewId};
use style_traits::CSSPixel;
use webrender_api::ExternalScrollId;
use webrender_api::units::{DevicePixel, DevicePoint, LayoutVector2D};

use self::TouchSequenceState::*;
use crate::native_scroll::{
    SNAP_DURATION_MS, ScrollDeceleration, ScrollTracking, begin_scroll_tracking,
    clamp_scroll_offset, ease_snap_progress, end_scroll_tracking, missed_scroll_frame_count,
    move_scroll_tracking, step_scroll_deceleration_frames,
};
use crate::paint::RepaintReason;
use crate::painter::Painter;
use crate::refresh_driver::{BaseRefreshDriver, RefreshDriverObserver};
use crate::webview_renderer::{ScrollZoomEvent, WebViewRenderer};

/// An ID for a sequence of touch events between a `Down` and the `Up` or `Cancel` event.
/// The ID is the same for all events between `Down` and `Up` or `Cancel`
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub(crate) struct TouchSequenceId(u32);

impl TouchSequenceId {
    const fn new() -> Self {
        Self(0)
    }

    /// Increments the ID for the next touch sequence.
    ///
    /// The increment is wrapping, since we can assume that the touch handler
    /// script for touch sequence N will have finished processing by the time
    /// we have wrapped around.
    fn next(&mut self) {
        self.0 = self.0.wrapping_add(1);
    }
}

const TOUCH_PINCH_MIN_SCREEN_PX: f32 = 5.0;

pub struct TouchHandler {
    /// The [`WebViewId`] of the `WebView` this [`TouchHandler`] is associated with.
    webview_id: WebViewId,
    pub current_sequence_id: TouchSequenceId,
    // todo: VecDeque + modulo arithmetic would be more efficient.
    touch_sequence_map: FxHashMap<TouchSequenceId, TouchSequenceInfo>,
    /// A set of [`InputEventId`]s for touch events that have been sent to the Constellation
    /// and have not been handled yet.
    pub(crate) pending_touch_input_events: RefCell<FxHashMap<InputEventId, PendingTouchInputEvent>>,
    /// This flag records whether the native scroll observer is active.
    observing_frames_for_native_scroll: Cell<bool>,
}

/// Whether the default move action is allowed or not.
#[derive(Debug, Eq, PartialEq)]
pub enum TouchMoveAllowed {
    /// The default move action is prevented by script
    Prevented,
    /// The default move action is allowed
    Allowed,
    /// The initial move handler result is still pending
    Pending,
}

pub(crate) enum TouchIdMoveTracking {
    Track,
    Remove,
}

/// The axis of a pan gesture. Once panning begins, the gesture is locked to the
/// dominant axis for the rest of the sequence, so that e.g. a vertical pan that
/// passes over a horizontally scrollable element keeps scrolling the page instead
/// of switching to horizontal scrolling.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PanAxis {
    Horizontal,
    Vertical,
}

/// The axis-lock policy for a pan gesture, decided at pan-start from the hit
/// node's `touch-action` and scrollable axes plus the gesture's dominant axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum PanPolicy {
    /// Not yet decided (the hit-test/scroll-tree lookup hasn't run). The first
    /// panning move emits the full 2D delta; the policy is set retroactively
    /// once the hit node is resolved.
    Undetermined,
    /// `touch-action: none` (or `pinch-zoom` alone): no single-finger direct
    /// manipulation. Suppress scrolling and fling.
    NoScroll,
    /// Lock the gesture to a single axis (zero the other axis in the emitted
    /// delta and the velocity). Used for `pan-x`/`pan-y`, and for `auto` when
    /// the hit node cannot scroll the dominant axis (scroll-chaining lock).
    Lock(PanAxis),
    /// No lock: emit the full 2D delta so both axes scroll freely. Used for
    /// `auto`/`manipulation`/`pan-x pan-y` when the hit node can scroll the
    /// dominant axis.
    Free,
}

impl PanPolicy {
    /// `NoScroll` is handled by the caller which suppresses the action entirely.
    /// Only here to keep the match exhaustive.
    fn pan_delta(self, delta: Vector2D<f32, DevicePixel>) -> Vector2D<f32, DevicePixel> {
        match self {
            PanPolicy::Lock(axis) => match axis {
                PanAxis::Horizontal => Vector2D::new(delta.x, 0.0),
                PanAxis::Vertical => Vector2D::new(0.0, delta.y),
            },
            PanPolicy::Free | PanPolicy::Undetermined | PanPolicy::NoScroll => delta,
        }
    }
}

/// Input captured at touch-down for deciding [`PanPolicy`] at pan-start.
/// `touch_action` and the structurally scrollable axes of the hit node.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PanPolicyInput {
    pub touch_action: TouchAction,
    pub scrollable_x: bool,
    pub scrollable_y: bool,
}

impl PanPolicyInput {
    /// Decide the [`PanPolicy`] for a gesture with the given dominant axis.
    /// See the policy table documented on [`PanPolicy`].
    fn to_pan_policy(self, dominant: PanAxis) -> PanPolicy {
        match self.touch_action {
            TouchAction::None => PanPolicy::NoScroll,
            TouchAction::PanX | TouchAction::PanY => PanPolicy::Lock(dominant),
            TouchAction::Auto => {
                if self.scrollable_x && self.scrollable_y {
                    PanPolicy::Free
                } else {
                    PanPolicy::Lock(dominant)
                }
            },
        }
    }
}

/// A cached [`PaintHitTestResult`] to use during a touch sequence. This
/// is kept so that the renderer doesn't have to constantly keep making hit tests
/// while during panning and flinging actions.
struct HitTestResultCache {
    value: PaintHitTestResult,
    device_pixels_per_page: Scale<f32, CSSPixel, DevicePixel>,
}

pub struct TouchSequenceInfo {
    /// touch sequence state
    pub(crate) state: TouchSequenceState,
    /// touch sequence active touch points
    active_touch_points: Vec<TouchPoint>,
    /// Whether the script thread is already processing a touchmove operation for the TouchId.
    ///
    /// We use this to skip sending the event to the script thread,
    /// to prevent overloading script.
    touch_ids_in_move: FxHashSet<TouchId>,
    /// Do not perform a click action.
    ///
    /// This happens when
    /// - We had a touch move larger than the minimum distance OR
    /// - We had multiple active touchpoints OR
    /// - `preventDefault()` was called in a touch_down or touch_up handler
    pub prevent_click: bool,
    /// Whether move is allowed, prevented or the result is still pending.
    /// Once the first move has been processed by script, we can transition to
    /// non-cancellable events, and directly perform the pan without waiting for script.
    pub prevent_move: TouchMoveAllowed,
    /// Move operation waiting to be processed in the touch sequence.
    ///
    /// This is only used while the first touch move is processed in script.
    /// Todo: It would be nice to merge this into the TouchSequenceState, but
    /// this requires some additional work to handle the merging of pending
    /// touch move events. Presumably if we keep a history of previous touch points,
    /// this would allow a better fling algorithm and easier merging of zoom events.
    pending_touch_move_actions: Vec<ScrollZoomEvent>,
    /// Cache for the last touch hit test result.
    hit_test_result_cache: FxHashMap<TouchId, HitTestResultCache>,
    sequence_started: Instant,
    pub(crate) pan_policy_input: Option<PanPolicyInput>,
}

impl TouchSequenceInfo {
    fn touch_count(&self) -> usize {
        self.active_touch_points.len()
    }

    fn pinch_distance_and_center(&self) -> (f32, Point2D<f32, DevicePixel>) {
        debug_assert_eq!(self.touch_count(), 2);
        let p0 = self.active_touch_points[0].point;
        let p1 = self.active_touch_points[1].point;
        let center = p0.lerp(p1, 0.5);
        let distance = (p0 - p1).length();

        (distance, center)
    }

    fn add_pending_touch_move_action(&mut self, action: ScrollZoomEvent) {
        debug_assert!(self.prevent_move == TouchMoveAllowed::Pending);
        self.pending_touch_move_actions.push(action);
    }

    /// Returns true when all touch events of a sequence have been received.
    /// This does not mean that all event handlers have finished yet.
    fn is_finished(&self) -> bool {
        matches!(
            self.state,
            Finished | ScrollingAnimation { .. } | PendingScrollAnimation { .. } | PendingClick(_)
        )
    }

    fn update_hit_test_result_cache_pointer(
        &mut self,
        touch_id: TouchId,
        delta: Vector2D<f32, DevicePixel>,
    ) {
        if let Some(hit_test_result_cache) = self.hit_test_result_cache.get_mut(&touch_id) {
            let scaled_delta = delta / hit_test_result_cache.device_pixels_per_page;
            // Update the point of the hit test result to match the current touch point.
            hit_test_result_cache.value.point_in_viewport += scaled_delta;
        }
    }
}

/// An action that can be immediately performed in response to a touch move event
/// without waiting for script.
#[derive(Clone, Copy, Debug, PartialEq)]

pub struct TouchPoint {
    pub touch_id: TouchId,
    pub point: Point2D<f32, DevicePixel>,
}

impl TouchPoint {
    fn new(touch_id: TouchId, point: Point2D<f32, DevicePixel>) -> Self {
        TouchPoint { touch_id, point }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct NativeScrollTarget {
    pub pipeline_id: PipelineId,
    pub external_scroll_id: ExternalScrollId,
    pub logical_offset: LayoutVector2D,
    pub maximum_offset: LayoutVector2D,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct NativeScrollAction {
    pub target: NativeScrollTarget,
    pub visual_offset: LayoutVector2D,
    pub finished: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct NativeScrollGesture {
    pub target: NativeScrollTarget,
    pub tracking: ScrollTracking,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum NativeScrollAnimation {
    Decelerating {
        state: ScrollDeceleration,
        last_frame: Instant,
    },
    Snapping {
        from: f64,
        to: f64,
        started: Instant,
    },
}

/// The states of the touch input state machine.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum TouchSequenceState {
    /// touch point is active but does not start moving
    Touching { scroll: Option<NativeScrollGesture> },
    /// A single touch point is active and has started panning.
    Panning { scroll: NativeScrollGesture },
    /// A two-finger pinch zoom gesture is active.
    Pinching,
    /// A multi-touch gesture is in progress.
    MultiTouch,
    // All states below here are reached after a touch-up, i.e. all events of the sequence
    // have already been received.
    /// The initial move handler must allow the native scroll animation.
    PendingScrollAnimation {
        target: NativeScrollTarget,
        animation: NativeScrollAnimation,
        cursor: DevicePoint,
    },
    /// No active touch points, but there is still scrolling velocity
    ScrollingAnimation {
        target: NativeScrollTarget,
        animation: NativeScrollAnimation,
        cursor: DevicePoint,
    },
    /// The touch sequence is finished, but a click is still pending, waiting on script.
    PendingClick(DevicePoint),
    /// touch sequence finished.
    Finished,
}

impl TouchHandler {
    pub(crate) fn new(webview_id: WebViewId) -> Self {
        let finished_info = TouchSequenceInfo {
            state: TouchSequenceState::Finished,
            active_touch_points: vec![],
            touch_ids_in_move: FxHashSet::default(),
            prevent_click: false,
            prevent_move: TouchMoveAllowed::Pending,
            pending_touch_move_actions: vec![],
            hit_test_result_cache: FxHashMap::default(),
            sequence_started: Instant::now(),
            pan_policy_input: None,
        };
        // We insert a simulated initial touch sequence, which is already finished,
        // so that we always have one element in the map, which simplifies creating
        // a new touch sequence on touch_down.
        let mut touch_sequence_map = FxHashMap::default();
        touch_sequence_map.insert(TouchSequenceId::new(), finished_info);
        TouchHandler {
            webview_id,
            current_sequence_id: TouchSequenceId::new(),
            touch_sequence_map,
            pending_touch_input_events: Default::default(),
            observing_frames_for_native_scroll: Default::default(),
        }
    }

    pub(crate) fn set_handling_touch_move_for_touch_id(
        &mut self,
        sequence_id: TouchSequenceId,
        touch_id: TouchId,
        flag: TouchIdMoveTracking,
    ) {
        if let Some(sequence) = self.touch_sequence_map.get_mut(&sequence_id) {
            match flag {
                TouchIdMoveTracking::Track => {
                    sequence.touch_ids_in_move.insert(touch_id);
                }
                TouchIdMoveTracking::Remove => {
                    sequence.touch_ids_in_move.remove(&touch_id);
                }
            }
        }
    }

    pub(crate) fn is_handling_touch_move_for_touch_id(
        &self,
        sequence_id: TouchSequenceId,
        touch_id: TouchId,
    ) -> bool {
        self.touch_sequence_map
            .get(&sequence_id)
            .is_some_and(|seq| seq.touch_ids_in_move.contains(&touch_id))
    }

    pub(crate) fn prevent_click(&mut self, sequence_id: TouchSequenceId) {
        if let Some(sequence) = self.touch_sequence_map.get_mut(&sequence_id) {
            sequence.prevent_click = true;
        } else {
            warn!("TouchSequenceInfo corresponding to the sequence number has been deleted.");
        }
    }

    pub(crate) fn prevent_move(&mut self, sequence_id: TouchSequenceId) {
        if let Some(sequence) = self.touch_sequence_map.get_mut(&sequence_id) {
            sequence.prevent_move = TouchMoveAllowed::Prevented;
        } else {
            warn!("TouchSequenceInfo corresponding to the sequence number has been deleted.");
        }
    }

    /// Returns true if default move actions are allowed, false if prevented or the result
    /// is still pending.,
    pub(crate) fn move_allowed(&self, sequence_id: TouchSequenceId) -> bool {
        self.touch_sequence_map
            .get(&sequence_id)
            .is_none_or(|sequence| sequence.prevent_move == TouchMoveAllowed::Allowed)
    }

    pub(crate) fn take_pending_touch_move_actions(
        &mut self,
        sequence_id: TouchSequenceId,
    ) -> Vec<ScrollZoomEvent> {
        self.touch_sequence_map
            .get_mut(&sequence_id)
            .map(|sequence| std::mem::take(&mut sequence.pending_touch_move_actions))
            .unwrap_or_default()
    }

    pub(crate) fn remove_pending_touch_move_actions(&mut self, sequence_id: TouchSequenceId) {
        if let Some(sequence) = self.touch_sequence_map.get_mut(&sequence_id) {
            sequence.pending_touch_move_actions.clear();
        }
    }

    // try to remove touch sequence, if touch sequence end and not has pending action.
    pub(crate) fn try_remove_touch_sequence(&mut self, sequence_id: TouchSequenceId) {
        if let Some(sequence) = self.touch_sequence_map.get(&sequence_id)
            && sequence.pending_touch_move_actions.is_empty()
            && sequence.state == Finished
        {
            self.touch_sequence_map.remove(&sequence_id);
        }
    }

    pub(crate) fn remove_touch_sequence(&mut self, sequence_id: TouchSequenceId) {
        let old = self.touch_sequence_map.remove(&sequence_id);
        debug_assert!(old.is_some(), "Sequence already removed?");
    }

    fn get_current_touch_sequence_mut(&mut self) -> &mut TouchSequenceInfo {
        self.touch_sequence_map
            .get_mut(&self.current_sequence_id)
            .expect("Current Touch sequence does not exist")
    }

    fn try_get_current_touch_sequence(&self) -> Option<&TouchSequenceInfo> {
        self.touch_sequence_map.get(&self.current_sequence_id)
    }

    fn try_get_current_touch_sequence_mut(&mut self) -> Option<&mut TouchSequenceInfo> {
        self.touch_sequence_map.get_mut(&self.current_sequence_id)
    }

    fn get_touch_sequence(&self, sequence_id: TouchSequenceId) -> &TouchSequenceInfo {
        self.touch_sequence_map
            .get(&sequence_id)
            .expect("Touch sequence not found.")
    }

    pub(crate) fn get_touch_sequence_mut(
        &mut self,
        sequence_id: TouchSequenceId,
    ) -> Option<&mut TouchSequenceInfo> {
        self.touch_sequence_map.get_mut(&sequence_id)
    }

    fn elapsed_ms(sequence_started: Instant, now: Instant) -> f64 {
        now.saturating_duration_since(sequence_started)
            .as_secs_f64()
            * 1000.0
    }

    fn native_scroll_action(
        target: NativeScrollTarget,
        content_offset: f64,
        finished: bool,
    ) -> NativeScrollAction {
        NativeScrollAction {
            target,
            visual_offset: LayoutVector2D::new(target.logical_offset.x, -(content_offset as f32)),
            finished,
        }
    }

    fn animation_offset(animation: NativeScrollAnimation, now: Instant) -> f64 {
        match animation {
            NativeScrollAnimation::Decelerating { state, .. } => state.offset,
            NativeScrollAnimation::Snapping { from, to, started } => {
                let elapsed_ms = now.saturating_duration_since(started).as_secs_f64() * 1000.0;
                from + (to - from) * ease_snap_progress(elapsed_ms)
            }
        }
    }

    fn resting_action_for_state(
        state: TouchSequenceState,
        now: Instant,
    ) -> Option<NativeScrollAction> {
        let (target, offset) = match state {
            Touching {
                scroll: Some(scroll),
            }
            | Panning { scroll } => (scroll.target, scroll.tracking.offset),
            PendingScrollAnimation {
                target, animation, ..
            }
            | ScrollingAnimation {
                target, animation, ..
            } => (target, Self::animation_offset(animation, now)),
            _ => return None,
        };
        let min_offset = -(target.maximum_offset.y as f64);
        Some(Self::native_scroll_action(
            target,
            clamp_scroll_offset(offset, min_offset),
            true,
        ))
    }

    pub(crate) fn on_touch_down(
        &mut self,
        touch_id: TouchId,
        point: Point2D<f32, DevicePixel>,
        target: Option<NativeScrollTarget>,
        scale: f32,
        now: Instant,
    ) -> Option<ScrollZoomEvent> {
        let interrupted_action = self.try_get_current_touch_sequence().and_then(|sequence| {
            Self::resting_action_for_state(sequence.state, now).map(ScrollZoomEvent::NativeScroll)
        });

        if !self
            .touch_sequence_map
            .contains_key(&self.current_sequence_id)
            || self
                .get_touch_sequence(self.current_sequence_id)
                .is_finished()
        {
            self.current_sequence_id.next();
            debug!("Entered new touch sequence: {:?}", self.current_sequence_id);
            if let Some(sequence) = self.try_get_current_touch_sequence_mut() {
                sequence.state = Finished;
            }
            self.observing_frames_for_native_scroll.set(false);
            let active_touch_points = vec![TouchPoint::new(touch_id, point)];
            let scroll = target.map(|target| NativeScrollGesture {
                target,
                tracking: begin_scroll_tracking(
                    -(target.logical_offset.y as f64),
                    -(target.maximum_offset.y as f64),
                    point.y as f64 / scale as f64,
                    0.0,
                ),
            });
            self.touch_sequence_map.insert(
                self.current_sequence_id,
                TouchSequenceInfo {
                    state: Touching { scroll },
                    active_touch_points,
                    touch_ids_in_move: FxHashSet::default(),
                    prevent_click: false,
                    prevent_move: TouchMoveAllowed::Pending,
                    pending_touch_move_actions: vec![],
                    hit_test_result_cache: FxHashMap::default(),
                    sequence_started: now,
                    pan_policy_input: None,
                },
            );
        } else {
            debug!("Touch down in sequence {:?}.", self.current_sequence_id);
            let touch_sequence = self.get_current_touch_sequence_mut();
            touch_sequence
                .active_touch_points
                .push(TouchPoint::new(touch_id, point));
            let restoration = Self::resting_action_for_state(touch_sequence.state, now)
                .map(ScrollZoomEvent::NativeScroll);
            match touch_sequence.active_touch_points.len() {
                2.. => {
                    touch_sequence.state = MultiTouch;
                }
                0..2 => {
                    unreachable!("Secondary touch_down event with less than 2 fingers active?");
                }
            }
            // Multiple fingers prevent a click.
            touch_sequence.prevent_click = true;
            return restoration;
        }
        interrupted_action
    }

    pub(crate) fn notify_new_frame_start(&mut self, now: Instant) -> Option<NativeScrollAction> {
        let touch_sequence = self.touch_sequence_map.get_mut(&self.current_sequence_id)?;
        let ScrollingAnimation {
            target,
            animation,
            cursor: _,
        } = touch_sequence.state
        else {
            self.observing_frames_for_native_scroll.set(false);
            return None;
        };

        match animation {
            NativeScrollAnimation::Decelerating { state, last_frame } => {
                let elapsed_ms = now.saturating_duration_since(last_frame).as_secs_f64() * 1000.0;
                let frame_count = missed_scroll_frame_count(elapsed_ms) + 1;
                let state = step_scroll_deceleration_frames(state, frame_count);
                if state.decelerating {
                    touch_sequence.state = ScrollingAnimation {
                        target,
                        animation: NativeScrollAnimation::Decelerating {
                            state,
                            last_frame: now,
                        },
                        cursor: DevicePoint::zero(),
                    };
                    return Some(Self::native_scroll_action(target, state.offset, false));
                }

                let resting_offset = clamp_scroll_offset(state.offset, state.min_offset);
                if resting_offset != state.offset {
                    touch_sequence.state = ScrollingAnimation {
                        target,
                        animation: NativeScrollAnimation::Snapping {
                            from: state.offset,
                            to: resting_offset,
                            started: now,
                        },
                        cursor: DevicePoint::zero(),
                    };
                    return Some(Self::native_scroll_action(target, state.offset, false));
                }

                touch_sequence.state = Finished;
                self.observing_frames_for_native_scroll.set(false);
                Some(Self::native_scroll_action(target, resting_offset, true))
            }
            NativeScrollAnimation::Snapping { from, to, started } => {
                let elapsed_ms = now.saturating_duration_since(started).as_secs_f64() * 1000.0;
                let finished = elapsed_ms >= SNAP_DURATION_MS;
                let offset = if finished {
                    to
                } else {
                    from + (to - from) * ease_snap_progress(elapsed_ms)
                };
                if finished {
                    touch_sequence.state = Finished;
                    self.observing_frames_for_native_scroll.set(false);
                }
                Some(Self::native_scroll_action(target, offset, finished))
            }
        }
    }

    pub(crate) fn stop_native_scroll_animation_if_needed(&mut self) {
        let current_sequence_id = self.current_sequence_id;
        let Some(touch_sequence) = self.try_get_current_touch_sequence_mut() else {
            return;
        };
        let ScrollingAnimation { .. } = touch_sequence.state else {
            return;
        };
        touch_sequence.state = Finished;
        self.try_remove_touch_sequence(current_sequence_id);
        self.observing_frames_for_native_scroll.set(false);
    }

    /// Whether a native scroll animation is currently observing frames. While
    /// this is true the painter must keep scheduling repaints: the animation
    /// only advances at frame starts, and a sub-pixel animation step leaves
    /// the WebRender frame unchanged, so no new-frame signal would arrive to
    /// schedule the next frame start and the animation would park mid-flight.
    pub(crate) fn has_ongoing_native_scroll_animation(&self) -> bool {
        self.observing_frames_for_native_scroll.get()
    }

    pub(crate) fn on_touch_move(
        &mut self,
        touch_id: TouchId,
        point: Point2D<f32, DevicePixel>,
        scale: f32,
        now: Instant,
    ) -> Option<ScrollZoomEvent> {
        // As `TouchHandler` is per `WebViewRenderer` which is per `WebView` we might get a Touch Sequence Move that
        // started with a down on a different webview. As the touch_sequence id is only changed on touch_down this
        // move event gets a touch id which is already cleaned up.
        let sequence_started = self.try_get_current_touch_sequence()?.sequence_started;
        let event_time = Self::elapsed_ms(sequence_started, now);
        let touch_sequence = self.try_get_current_touch_sequence_mut()?;
        let idx = match touch_sequence
            .active_touch_points
            .iter_mut()
            .position(|t| t.touch_id == touch_id)
        {
            Some(i) => i,
            None => {
                error!("Got a touchmove event for a non-active touch point");
                return None;
            }
        };
        let old_point = touch_sequence.active_touch_points[idx].point;
        let delta = point - old_point;
        touch_sequence.update_hit_test_result_cache_pointer(touch_id, delta);

        let action = match touch_sequence.touch_count() {
            1 => match touch_sequence.state {
                Panning { mut scroll } => {
                    let moved = move_scroll_tracking(
                        scroll.tracking,
                        point.y as f64 / scale as f64,
                        event_time,
                    );
                    scroll.tracking = moved.state;
                    touch_sequence.state = Panning { scroll };
                    touch_sequence.active_touch_points[idx].point = point;
                    Some(ScrollZoomEvent::NativeScroll(Self::native_scroll_action(
                        scroll.target,
                        scroll.tracking.offset,
                        false,
                    )))
                }
                Touching {
                    scroll: Some(mut scroll),
                } => {
                    let dominant = if delta.y.abs() > delta.x.abs() {
                        PanAxis::Vertical
                    } else {
                        PanAxis::Horizontal
                    };
                    let policy = touch_sequence.pan_policy_input
                        .map(|input| input.to_pan_policy(dominant))
                        .unwrap_or(PanPolicy::Undetermined);
                    if policy == PanPolicy::NoScroll || policy.pan_delta(delta).y == 0.0 {
                        return None;
                    }
                    let moved = move_scroll_tracking(
                        scroll.tracking,
                        point.y as f64 / scale as f64,
                        event_time,
                    );
                    scroll.tracking = moved.state;
                    if moved.began_dragging {
                        touch_sequence.state = Panning { scroll };
                        touch_sequence.prevent_click = true;
                        touch_sequence.active_touch_points[idx].point = point;
                    } else {
                        touch_sequence.state = Touching {
                            scroll: Some(scroll),
                        };
                    }
                    None
                }
                _ => None,
            },
            2 => {
                if touch_sequence.state == Pinching
                    || delta.x.abs() > TOUCH_PINCH_MIN_SCREEN_PX * scale
                    || delta.y.abs() > TOUCH_PINCH_MIN_SCREEN_PX * scale
                {
                    touch_sequence.state = Pinching;
                    let (d0, _) = touch_sequence.pinch_distance_and_center();

                    // update the touch point with the enough distance or pinching.
                    touch_sequence.active_touch_points[idx].point = point;
                    let (d1, c1) = touch_sequence.pinch_distance_and_center();

                    Some(ScrollZoomEvent::PinchZoom(d1 / d0, c1))
                } else {
                    // We don't update the touchpoint, so multiple small moves can
                    // accumulate and merge into a larger move.
                    None
                }
            }
            _ => {
                touch_sequence.active_touch_points[idx].point = point;
                touch_sequence.state = MultiTouch;
                None
            }
        };
        // If the first move has not been processed yet, buffer the action.
        if let Some(action) = action &&
            touch_sequence.prevent_move == TouchMoveAllowed::Pending
        {
            touch_sequence.add_pending_touch_move_action(action);
        }

        action
    }

    pub(crate) fn on_touch_up(
        &mut self,
        touch_id: TouchId,
        point: Point2D<f32, DevicePixel>,
        scale: f32,
        now: Instant,
    ) -> Option<ScrollZoomEvent> {
        let sequence_started = self.try_get_current_touch_sequence()?.sequence_started;
        let event_time = Self::elapsed_ms(sequence_started, now);
        let Some(touch_sequence) = self.try_get_current_touch_sequence_mut() else {
            warn!("Current touch sequence not found");
            return None;
        };
        match touch_sequence
            .active_touch_points
            .iter()
            .position(|t| t.touch_id == touch_id)
        {
            Some(i) => {
                touch_sequence.active_touch_points.swap_remove(i);
            }
            None => {
                warn!("Got a touchup event for a non-active touch point");
                return None;
            }
        };
        let action = match touch_sequence.state {
            Touching { .. } => {
                if touch_sequence.prevent_click {
                    touch_sequence.state = Finished;
                } else {
                    touch_sequence.state = PendingClick(point);
                }
                None
            }
            Panning { scroll } => {
                let release = end_scroll_tracking(
                    scroll.tracking,
                    event_time,
                    Some(point.y as f64 / scale as f64),
                );
                let target = scroll.target;
                let visual_action = ScrollZoomEvent::NativeScroll(Self::native_scroll_action(
                    target,
                    release.state.offset,
                    false,
                ));
                let animation = if release.deceleration.decelerating {
                    Some(NativeScrollAnimation::Decelerating {
                        state: release.deceleration,
                        last_frame: now,
                    })
                } else {
                    let resting_offset =
                        clamp_scroll_offset(release.state.offset, release.state.min_offset);
                    (resting_offset != release.state.offset).then_some(
                        NativeScrollAnimation::Snapping {
                            from: release.state.offset,
                            to: resting_offset,
                            started: now,
                        },
                    )
                };
                if let Some(animation) = animation {
                    touch_sequence.state = match touch_sequence.prevent_move {
                        TouchMoveAllowed::Allowed => ScrollingAnimation {
                            target,
                            animation,
                            cursor: point,
                        },
                        TouchMoveAllowed::Pending => PendingScrollAnimation {
                            target,
                            animation,
                            cursor: point,
                        },
                        TouchMoveAllowed::Prevented => Finished,
                    };
                } else {
                    touch_sequence.state = Finished;
                }
                Some(visual_action)
            }
            Pinching => {
                touch_sequence.state = Touching { scroll: None };
                None
            }
            MultiTouch => {
                if touch_sequence.active_touch_points.is_empty() {
                    touch_sequence.state = Finished;
                }
                None
            }
            PendingScrollAnimation { .. }
            | ScrollingAnimation { .. }
            | PendingClick(_)
            | Finished => {
                error!("Touch-up received after the touch sequence ended.");
                None
            }
        };
        if let Some(action) = action
            && touch_sequence.prevent_move == TouchMoveAllowed::Pending
        {
            touch_sequence.add_pending_touch_move_action(action);
        }
        #[cfg(debug_assertions)]
        if touch_sequence.active_touch_points.is_empty() {
            debug_assert!(
                touch_sequence.is_finished(),
                "Did not transition to a finished state: {:?}",
                touch_sequence.state
            );
        }
        debug!(
            "Touch up with remaining active touchpoints: {:?}, in sequence {:?}",
            touch_sequence.active_touch_points.len(),
            self.current_sequence_id
        );
        action
    }

    pub(crate) fn on_touch_cancel(
        &mut self,
        touch_id: TouchId,
        _point: Point2D<f32, DevicePixel>,
        now: Instant,
    ) -> Option<ScrollZoomEvent> {
        let Some(touch_sequence) = self.try_get_current_touch_sequence_mut() else {
            return None;
        };
        let restoration = Self::resting_action_for_state(touch_sequence.state, now)
            .map(ScrollZoomEvent::NativeScroll);
        match touch_sequence
            .active_touch_points
            .iter()
            .position(|t| t.touch_id == touch_id)
        {
            Some(i) => {
                touch_sequence.active_touch_points.swap_remove(i);
            }
            None => {
                warn!("Got a touchcancel event for a non-active touch point");
                return None;
            }
        }
        if touch_sequence.active_touch_points.is_empty() {
            touch_sequence.state = Finished;
        }
        restoration
    }

    pub(crate) fn primary_touch_id(&self) -> Option<TouchId> {
        self.try_get_current_touch_sequence()?
            .active_touch_points
            .first()
            .map(|touch| touch.touch_id)
    }

    pub(crate) fn get_hit_test_result_cache_value(
        &self,
        touch_id: TouchId,
    ) -> Option<PaintHitTestResult> {
        let sequence = self.touch_sequence_map.get(&self.current_sequence_id)?;
        if sequence.state == Finished {
            return None;
        }
        sequence
            .hit_test_result_cache
            .get(&touch_id)
            .map(|cache| cache.value.clone())
    }

    pub(crate) fn set_hit_test_result_cache_value(
        &mut self,
        touch_id: TouchId,
        value: PaintHitTestResult,
        device_pixels_per_page: Scale<f32, CSSPixel, DevicePixel>,
    ) {
        if let Some(sequence) = self.touch_sequence_map.get_mut(&self.current_sequence_id) {
            sequence.hit_test_result_cache.entry(touch_id).or_insert(HitTestResultCache {
                value,
                device_pixels_per_page,
            });
        }
    }

    /// Capture the [`PanPolicyInput`] for the current touch sequence, from the
    /// hit node resolved at touch-down. Used by the renderer to feed the
    /// `touch-action` + scrollable axes of the hit node into [`PanPolicy`]
    /// decision at pan-start.
    pub(crate) fn set_pan_policy_input(&mut self, input: PanPolicyInput) {
        if let Some(sequence) = self.touch_sequence_map.get_mut(&self.current_sequence_id) {
            sequence.pan_policy_input = Some(input);
        }
    }

    pub(crate) fn clear_hit_test_result_cache_value(&mut self, touch_id: TouchId) {
        if let Some(sequence) = self.touch_sequence_map.get_mut(&self.current_sequence_id) {
            sequence.hit_test_result_cache.remove(&touch_id);
        }
    }

    pub(crate) fn add_pending_touch_input_event(
        &self,
        id: InputEventId,
        touch_id: TouchId,
        event_type: TouchEventType,
    ) {
        self.pending_touch_input_events.borrow_mut().insert(
            id,
            PendingTouchInputEvent {
                event_type,
                sequence_id: self.current_sequence_id,
                touch_id,
            },
        );
    }

    pub(crate) fn take_pending_touch_input_event(
        &self,
        id: InputEventId,
    ) -> Option<PendingTouchInputEvent> {
        self.pending_touch_input_events.borrow_mut().remove(&id)
    }

    pub(crate) fn add_touch_move_refresh_observer_if_necessary(
        &self,
        refresh_driver: Rc<BaseRefreshDriver>,
        repaint_reason: &Cell<RepaintReason>,
    ) {
        if self.observing_frames_for_native_scroll.get() {
            return;
        }

        let Some(current_touch_sequence) = self.try_get_current_touch_sequence() else {
            return;
        };

        if !matches!(
            current_touch_sequence.state,
            TouchSequenceState::ScrollingAnimation { .. },
        ) {
            return;
        }

        refresh_driver.add_observer(Rc::new(NativeScrollRefreshDriverObserver {
            webview_id: self.webview_id,
        }));
        self.observing_frames_for_native_scroll.set(true);
        repaint_reason.set(repaint_reason.get().union(RepaintReason::StartedFlinging));
    }
}

/// This data structure is used to store information about touch events that are
/// sent from the Renderer to the Constellation, so that they can finish processing
/// once their DOM events are fired.
pub(crate) struct PendingTouchInputEvent {
    pub event_type: TouchEventType,
    pub sequence_id: TouchSequenceId,
    pub touch_id: TouchId,
}

pub(crate) struct NativeScrollRefreshDriverObserver {
    pub webview_id: WebViewId,
}

impl RefreshDriverObserver for NativeScrollRefreshDriverObserver {
    fn frame_started(&self, painter: &mut Painter) -> bool {
        painter
            .webview_renderer_mut(self.webview_id)
            .is_some_and(WebViewRenderer::update_touch_handling_at_new_frame_start)
    }
}

#[cfg(test)]
mod native_scroll_spring_tests {
    use std::cell::LazyCell;
    use std::time::Duration;

    use embedder_traits::{EventLoopWaker, RefreshDriver as EmbedderRefreshDriver};
    use servo_base::id::{PipelineNamespace, PipelineNamespaceId, TEST_PAINTER_ID};
    use webrender_api::ExternalScrollId;

    use super::*;
    use crate::refresh_driver::TimerRefreshDriver;

    struct NoopWaker;

    impl EventLoopWaker for NoopWaker {
        fn clone_box(&self) -> Box<dyn EventLoopWaker> {
            Box::new(NoopWaker)
        }
        fn wake(&self) {}
    }

    struct NoopRefreshDriver;

    impl EmbedderRefreshDriver for NoopRefreshDriver {
        fn observe_next_frame(&self, _start_frame_callback: Box<dyn Fn() + Send + 'static>) {}
    }

    fn install_namespace() {
        // The ids need a namespace on each thread. A second install on the same
        // thread panics, so each test thread installs at most once.
        thread_local! {
            static INSTALLED: Cell<bool> = const { Cell::new(false) };
        }
        INSTALLED.with(|installed| {
            if !installed.get() {
                PipelineNamespace::install(PipelineNamespaceId(0));
                installed.set(true);
            }
        });
    }

    fn make_refresh_driver() -> Rc<BaseRefreshDriver> {
        let timer: LazyCell<Rc<TimerRefreshDriver>> =
            LazyCell::new(|| Rc::new(TimerRefreshDriver::default()));
        Rc::new(BaseRefreshDriver::new(
            Box::new(NoopWaker),
            Some(Rc::new(NoopRefreshDriver)),
            &timer,
        ))
    }

    fn make_target() -> NativeScrollTarget {
        let pipeline_id = PipelineId::new();
        NativeScrollTarget {
            pipeline_id,
            external_scroll_id: ExternalScrollId(1, pipeline_id.into()),
            // The content is taller than the viewport. The list can scroll.
            logical_offset: LayoutVector2D::new(0.0, 0.0),
            maximum_offset: LayoutVector2D::new(0.0, 300.0),
        }
    }

    #[test]
    fn touch_hit_test_caches_are_isolated_and_reusable() {
        install_namespace();
        let mut handler = TouchHandler::new(WebViewId::new(TEST_PAINTER_ID));
        let first = TouchId(0);
        let second = TouchId(1);
        let start = Instant::now();
        let target = make_target();
        let hit = |x| PaintHitTestResult {
            pipeline_id: target.pipeline_id,
            point_in_viewport: Point2D::new(x, 10.0),
            external_scroll_id: target.external_scroll_id,
        };
        let points = |handler: &TouchHandler| {
            [first, second].map(|id| {
                handler
                    .get_hit_test_result_cache_value(id)
                    .unwrap()
                    .point_in_viewport
            })
        };

        handler.on_touch_down(first, Point2D::new(10.0, 10.0), None, 1.0, start);
        handler.on_touch_down(second, Point2D::new(50.0, 10.0), None, 1.0, start);
        for (touch_id, x) in [(first, 10.0), (second, 50.0)] {
            handler.set_hit_test_result_cache_value(touch_id, hit(x), Scale::new(1.0));
        }

        handler.on_touch_move(
            first,
            Point2D::new(20.0, 10.0),
            1.0,
            start + Duration::from_millis(16),
        );
        assert_eq!(
            points(&handler),
            [Point2D::new(20.0, 10.0), Point2D::new(50.0, 10.0)]
        );

        handler.clear_hit_test_result_cache_value(first);
        assert!(handler.get_hit_test_result_cache_value(first).is_none());
        handler.set_hit_test_result_cache_value(first, hit(80.0), Scale::new(1.0));
        assert_eq!(
            points(&handler),
            [Point2D::new(80.0, 10.0), Point2D::new(50.0, 10.0)]
        );
    }

    /// Drags past the top edge and releases. The release happens while the DOM
    /// has not yet answered the touch move. This is the stuck-spring case.
    fn overscroll_then_release() -> (TouchHandler, TouchSequenceId) {
        install_namespace();
        let mut handler = TouchHandler::new(WebViewId::new(TEST_PAINTER_ID));
        let touch_id = TouchId(0);
        let start = Instant::now();

        handler.on_touch_down(
            touch_id,
            Point2D::new(100.0, 400.0),
            Some(make_target()),
            1.0,
            start,
        );
        // Drag down past the top edge. This overscrolls and stretches the band.
        for step in 1..=6 {
            let millis = 16 * step as u64;
            handler.on_touch_move(
                touch_id,
                Point2D::new(100.0, 400.0 + 20.0 * step as f32),
                1.0,
                start + Duration::from_millis(millis),
            );
        }
        let sequence_id = handler.current_sequence_id;
        handler.on_touch_up(
            touch_id,
            Point2D::new(100.0, 520.0),
            1.0,
            start + Duration::from_millis(200),
        );
        (handler, sequence_id)
    }

    #[test]
    fn a_release_while_the_move_is_pending_parks_the_spring() {
        let (handler, _sequence_id) = overscroll_then_release();
        let info = handler.try_get_current_touch_sequence().unwrap();
        assert_eq!(
            info.prevent_move,
            TouchMoveAllowed::Pending,
            "the DOM has not answered the move yet"
        );
        let PendingScrollAnimation { animation, .. } = info.state else {
            panic!(
                "the release must park the snap in PendingScrollAnimation, got {:?}",
                info.state
            );
        };
        assert!(
            matches!(animation, NativeScrollAnimation::Snapping { .. }),
            "the parked animation must be the spring back, got {animation:?}"
        );
        let repaint_reason = Cell::new(RepaintReason::empty());
        handler.add_touch_move_refresh_observer_if_necessary(make_refresh_driver(), &repaint_reason);
        assert!(
            !handler.observing_frames_for_native_scroll.get(),
            "a parked spring must not be advancing yet"
        );
    }

    /// The painter polls this while scheduling repaints, because a sub-pixel
    /// spring step leaves the WebRender frame unchanged and no new-frame
    /// signal arrives to schedule the next frame start. The resolution below
    /// is what `WebViewRenderer::on_touch_event_processed` does when the DOM
    /// allows the move. The report must turn on the moment the spring starts,
    /// stay on until the animation finishes, and then turn off so the repaint
    /// requests stop.
    #[test]
    fn a_running_spring_reports_ongoing_until_it_finishes() {
        let (mut handler, sequence_id) = overscroll_then_release();

        let info = handler.get_touch_sequence_mut(sequence_id).unwrap();
        info.prevent_move = TouchMoveAllowed::Allowed;
        if let PendingScrollAnimation {
            target,
            animation,
            cursor,
        } = info.state
        {
            info.state = ScrollingAnimation {
                target,
                animation,
                cursor,
            };
        }
        let repaint_reason = Cell::new(RepaintReason::empty());
        handler.add_touch_move_refresh_observer_if_necessary(make_refresh_driver(), &repaint_reason);
        assert!(
            handler.has_ongoing_native_scroll_animation(),
            "a running spring must request repaints"
        );

        let mut now = Instant::now();
        for _ in 0..600 {
            now += Duration::from_millis(16);
            if handler.notify_new_frame_start(now).is_none() {
                break;
            }
        }
        assert!(
            !handler.has_ongoing_native_scroll_animation(),
            "a finished spring must stop requesting repaints"
        );
    }
}
