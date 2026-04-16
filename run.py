import cv2
import json
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO

print("Starting")

# ── load model & video ────────────────────────────────────────────────────────
model = YOLO("/home/huangchs/504/new/v12/runs/detect/train/weights/best.pt")
cap = cv2.VideoCapture("/home/huangchs/504/new/v12/4-corner.mp4")

fps = int(cap.get(cv2.CAP_PROP_FPS))
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter("output3.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# ── load regions ──────────────────────────────────────────────────────────────
with open("selected_regions.json") as f:
    regions = json.load(f)

stop_lines     = regions.get("stop_lines", [])
traffic_lights = regions.get("traffic_lights", [])  # list of dicts with quad + bulb dots
STOP_ZONES     = regions.get("stop_zones", [])
STOP_SIGNS     = regions.get("stop_signs", [])
ROIS           = regions.get("rois", [])
BULB_RADIUS    = regions.get("bulb_radius", 8)

# ── build lanes (traffic_light[i] <-> stop_line[i]) ──────────────────────────
num_lanes = max(len(stop_lines), len(traffic_lights), 1)
lanes = []
for i in range(num_lanes):
    lanes.append({
        "traffic_light": traffic_lights[i] if i < len(traffic_lights) else None,
        "stop_line":     stop_lines[i]     if i < len(stop_lines)     else None,
        "light_history": deque(maxlen=10),
    })

# ── state ─────────────────────────────────────────────────────────────────────
track_history       = defaultdict(list)
red_violators       = {}                # track_id -> frame_idx
stop_sign_timers    = {}                # track_id -> entry dict
stop_sign_complied  = set()
stop_sign_violators = set()

STILL_THRESHOLD  = 3
STOP_MIN_SECONDS = 1
STILL_WINDOW     = 10
STILL_STD_THRESH = 2.0
STOP_GRACE_FRAMES = 15

frame_idx = 0

LIGHT_COLORS = {
    "red":     (0,   0,   255),
    "yellow":  (0,   215, 255),
    "green":   (0,   255, 0),
    "unknown": (128, 128, 128),
}

BULB_COLORS = {
    "red":    (0,   0,   255),
    "yellow": (0,   215, 255),
    "green":  (0,   255, 0),
}

# ── helpers ───────────────────────────────────────────────────────────────────
def point_in_quad(px, py, quad):
    pts = np.array(quad, dtype=np.int32)
    return cv2.pointPolygonTest(pts, (float(px), float(py)), False) >= 0


def in_any_roi(cx, cy, rois):
    if not rois:
        return True
    return any(point_in_quad(cx, cy, quad) for quad in rois)


def draw_quad(frame, quad, color, label=None):
    pts = np.array(quad, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], True, color, 2)
    if label:
        cv2.putText(frame, label, tuple(quad[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_trail(frame, pts, color):
    for i in range(1, len(pts)):
        p1 = (int(pts[i-1][1]), int(pts[i-1][2]))
        p2 = (int(pts[i][1]),   int(pts[i][2]))
        cv2.line(frame, p1, p2, color, 2)


def get_bulb_state(frame, center):
    """Returns (brightness, hue_valid) for a bulb dot."""
    if center is None:
        return 0.0, False
    cx, cy = int(center[0]), int(center[1])
    r = BULB_RADIUS
    y0 = max(0, cy - r)
    y1 = min(frame.shape[0], cy + r)
    x0 = max(0, cx - r)
    x1 = min(frame.shape[1], cx + r)
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0, False

    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (cx - x0, cy - y0), r, 255, -1)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    brightness = cv2.mean(hsv[:, :, 2], mask=mask)[0]

    # mean hue and saturation inside the bulb
    mean_h = cv2.mean(hsv[:, :, 0], mask=mask)[0]
    mean_s = cv2.mean(hsv[:, :, 1], mask=mask)[0]

    return brightness, mean_h, mean_s


BULB_HUE_RANGES = {
    "red":    [(0, 10), (170, 180)],   # wraps around 0
    "yellow": [(15, 35)],
    "green":  [(40, 90)],
}
BULB_MIN_SATURATION = 60   # below this = washed out / white glare, not a real color


def hue_valid(mean_h, mean_s, bulb):
    if mean_s < BULB_MIN_SATURATION:
        return False
    ranges = BULB_HUE_RANGES[bulb]
    return any(lo <= mean_h <= hi for lo, hi in ranges)


def get_light_state(frame, tl):
    brightness = {}
    hue_ok     = {}

    for bulb in ["red", "yellow", "green"]:
        center = tl.get(bulb)
        if center is None:
            continue
        b, mean_h, mean_s = get_bulb_state(frame, center)
        brightness[bulb] = b
        hue_ok[bulb]     = b > 10 and hue_valid(mean_h, mean_s, bulb)

    if not brightness:
        return "unknown"

    # primary: brightest bulb that passes hue check
    hue_valid_bulbs = {k: v for k, v in brightness.items() if hue_ok[k]}
    if hue_valid_bulbs:
        return max(hue_valid_bulbs, key=hue_valid_bulbs.get)

    # fallback: just brightest bulb regardless of hue
    best = max(brightness, key=brightness.get)
    return best if brightness[best] > 10 else "unknown"


def smoothed_light_state(frame, lane):
    if lane["traffic_light"] is None:
        return "unknown"
    state = get_light_state(frame, lane["traffic_light"])
    lane["light_history"].append(state)
    return max(set(lane["light_history"]), key=lane["light_history"].count)


def crossed_stop_line(prev_pt, curr_pt, p1, p2):
    def side(px, py):
        return (p2[0]-p1[0])*(py-p1[1]) - (p2[1]-p1[1])*(px-p1[0])

    s1 = side(prev_pt[1], prev_pt[2])
    s2 = side(curr_pt[1], curr_pt[2])

    if (s1 < 0) == (s2 < 0):
        return False

    t = s1 / (s1 - s2)
    cross_x = prev_pt[1] + t * (curr_pt[1] - prev_pt[1])
    cross_y = prev_pt[2] + t * (curr_pt[2] - prev_pt[2])

    min_x, max_x = min(p1[0], p2[0]), max(p1[0], p2[0])
    min_y, max_y = min(p1[1], p2[1]), max(p1[1], p2[1])

    return (min_x <= cross_x <= max_x) and (min_y <= cross_y <= max_y)


def get_vehicle_lane(cx, lanes):
    best, best_dist = None, float("inf")
    for i, lane in enumerate(lanes):
        if lane["stop_line"] is None:
            continue
        p1, p2 = lane["stop_line"]
        mx = (p1[0] + p2[0]) / 2
        dist = abs(cx - mx)
        if dist < best_dist:
            best_dist = dist
            best = i
    return best


def has_stopped_in_zone(entry, fps):
    min_stop_frames = max(1, int(round(fps * STOP_MIN_SECONDS)))
    return entry["stationary_frames"] >= min_stop_frames


def is_stationary(positions):
    if len(positions) < STILL_WINDOW:
        return False
    recent = positions[-STILL_WINDOW:]
    xs = [p[0] for p in recent]
    ys = [p[1] for p in recent]
    return np.std(xs) < STILL_STD_THRESH and np.std(ys) < STILL_STD_THRESH


def check_stop_sign(track_id, cx, cy, frame_idx, fps):
    in_zone = any(point_in_quad(cx, cy, quad) for quad in STOP_ZONES)

    if not in_zone:
        if track_id not in stop_sign_timers:
            return
        entry = stop_sign_timers[track_id]
        if "exit_frame" not in entry:
            entry["exit_frame"] = frame_idx
        frames_outside = frame_idx - entry["exit_frame"]
        if frames_outside < STOP_GRACE_FRAMES:
            return
        stop_sign_timers.pop(track_id)
        if has_stopped_in_zone(entry, fps):
            stop_sign_complied.add(track_id)
        elif track_id not in stop_sign_complied:
            stop_sign_violators.add(track_id)
        return

    if track_id in stop_sign_timers:
        stop_sign_timers[track_id].pop("exit_frame", None)
    else:
        stop_sign_timers[track_id] = {
            "enter_frame":      frame_idx,
            "last_frame":       frame_idx,
            "last_pos":         (cx, cy),
            "positions":        [],
            "stationary_frames": 0,
            "is_stationary":    False,
        }

    entry = stop_sign_timers[track_id]
    entry["last_frame"] = frame_idx
    entry["positions"].append((cx, cy))
    entry["is_stationary"] = is_stationary(entry["positions"])
    if entry["is_stationary"]:
        entry["stationary_frames"] += 1
    else:
        entry["stationary_frames"] = 0
    entry["last_pos"] = (cx, cy)

    if track_id not in stop_sign_complied and has_stopped_in_zone(entry, fps):
        stop_sign_complied.add(track_id)


def draw_bulb_dots(frame, tl, active_state):
    """Draw bulb dots on the traffic light quad, highlight the active one."""
    for bulb, color in BULB_COLORS.items():
        center = tl.get(bulb)
        if center is None:
            continue
        cx, cy = int(center[0]), int(center[1])
        is_active = (bulb == active_state)
        thickness = -1 if is_active else 1
        radius    = BULB_RADIUS if is_active else max(BULB_RADIUS - 3, 2)
        cv2.circle(frame, (cx, cy), radius, color, thickness)
        cv2.circle(frame, (cx, cy), radius, (255, 255, 255), 1)

CROSS_LOOKBACK = 5  # frames to look back for crossing check

def check_crossed_stop_line(pts, p1, p2):
    """Check any consecutive pair in the last N points for a crossing."""
    recent = pts[-CROSS_LOOKBACK:]
    for i in range(1, len(recent)):
        if crossed_stop_line(recent[i-1], recent[i], p1, p2):
            return True
    return False

# ── main loop ─────────────────────────────────────────────────────────────────
cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tracking", 1920, 1080)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # stale tracker cleanup
    STALE_FRAMES = fps * 3
    stale_ids = [
        tid for tid, entry in stop_sign_timers.items()
        if frame_idx - entry["last_frame"] > STALE_FRAMES
    ]
    for tid in stale_ids:
        entry = stop_sign_timers.pop(tid)
        if has_stopped_in_zone(entry, fps):
            stop_sign_complied.add(tid)
        elif tid not in stop_sign_complied:
            stop_sign_violators.add(tid)

    # per-lane light states
    lane_states = [smoothed_light_state(frame, lane) for lane in lanes]

    # draw lane annotations
    for i, lane in enumerate(lanes):
        state = lane_states[i]
        color = LIGHT_COLORS[state]

        if lane["stop_line"]:
            p1 = tuple(lane["stop_line"][0])
            p2 = tuple(lane["stop_line"][1])
            cv2.line(frame, p1, p2, color, 2)
            cv2.putText(frame, f"L{i} STOP", p1,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if lane["traffic_light"]:
            tl = lane["traffic_light"]
            draw_quad(frame, tl["quad"], color, f"TL{i}: {state.upper()}")
            draw_bulb_dots(frame, tl, state)

    # draw stop zones
    for i, zone in enumerate(STOP_ZONES):
        draw_quad(frame, zone, (0, 255, 255), f"ZONE {i}")

    # draw ROIs
    for i, roi in enumerate(ROIS):
        draw_quad(frame, roi, (200, 200, 200), f"ROI {i}")

    # hud
    cv2.putText(frame, f"Frame: {frame_idx}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(frame, f"Red Violators: {len(red_violators)}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Stop Sign Violations: {len(stop_sign_violators)}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # run tracker
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.25, iou=0.4)
    r = results[0]

    if r.boxes.id is not None:
        ids   = r.boxes.id.cpu().numpy().astype(int)
        boxes = r.boxes.xyxy.cpu().numpy()

        for track_id, box in zip(ids, boxes):
            x1, y1, x2, y2 = box
            cx = float((x1 + x2) / 2.0)
            cy = float(y2)

            # 1. ROI filter
            if not in_any_roi(cx, cy, ROIS):
                continue

            # 2. update track history
            track_history[track_id].append((frame_idx, cx, cy))
            track_history[track_id] = track_history[track_id][-100:]
            pts = track_history[track_id]

            # 3. red light violation — assign to nearest lane
            lane_idx = get_vehicle_lane(cx, lanes)
            if lane_idx is not None:
                light_state = lane_states[lane_idx]
                lane        = lanes[lane_idx]

                if len(pts) >= 2 and light_state == "red" and lane["stop_line"]:
                    p1 = tuple(lane["stop_line"][0])
                    p2 = tuple(lane["stop_line"][1])
                    if check_crossed_stop_line(pts, p1, p2):
                        if track_id not in red_violators:
                            red_violators[track_id] = frame_idx
                            print(f"[frame {frame_idx}] RED LIGHT VIOLATION: track {track_id} lane {lane_idx}")
            else:
                light_state = "unknown"

            # 4. stop sign compliance (independent — uses stop zones)
            check_stop_sign(track_id, cx, cy, frame_idx, fps)

            # 5. draw
            is_red_violator  = track_id in red_violators
            is_stop_violator = track_id in stop_sign_violators
            is_any_violator  = is_red_violator or is_stop_violator
            color = (0, 0, 255) if is_any_violator else (0, 255, 0)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"ID {track_id}", (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            label_y = int(y2) + 20
            if is_red_violator:
                cv2.putText(frame, "RED LIGHT VIOLATION", (int(x1), label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                label_y += 20
            if is_stop_violator:
                cv2.putText(frame, "STOP SIGN VIOLATION", (int(x1), label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                label_y += 20
            elif track_id in stop_sign_complied:
                cv2.putText(frame, "STOP OK", (int(x1), label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                label_y += 20
            elif track_id in stop_sign_timers:
                entry     = stop_sign_timers[track_id]
                stop_secs = entry["stationary_frames"] / fps if fps > 0 else 0.0
                if entry["is_stationary"]:
                    text  = f"STOPPING {stop_secs:.1f}s"
                    tcol  = (0, 215, 255)
                else:
                    text  = f"MOVING stop={stop_secs:.1f}s"
                    tcol  = (0, 165, 255)
                cv2.putText(frame, text, (int(x1), label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, tcol, 2)

            draw_trail(frame, pts, color)

    out.write(frame)
    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

# ── cleanup ───────────────────────────────────────────────────────────────────
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\nDone.")
print(f"Red light violators: {len(red_violators)}")
for tid, fidx in red_violators.items():
    print(f"  Track {tid} at frame {fidx}")
print(f"Stop sign violators: {len(stop_sign_violators)}")
print(f"Stop sign complied:  {len(stop_sign_complied)}")