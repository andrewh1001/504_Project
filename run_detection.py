import cv2
import json
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO

print("Starting")
video_path = "bell3"
model = YOLO("/runs/detect/best/weights/best.pt")
cap = cv2.VideoCapture(f"/files/{video_path}.mp4")

fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out_cam = cv2.VideoWriter("output_cam.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
sat_img = cv2.imread(f"{video_path}.png")
if sat_img is None:
    raise RuntimeError(f"Could not load {video_path}.png")
out_sat = cv2.VideoWriter("output_sat.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (sat_img.shape[1], sat_img.shape[0]))

with open(f"homography_{video_path}.json") as f:
    hdata = json.load(f)
H = np.float32(hdata["H"])
print(f"Loaded homography")

with open(f"selected_regions_{video_path}.json") as f:
    regions = json.load(f)
print(f"Loaded regions")

stop_lines     = regions.get("stop_lines", [])
traffic_lights = regions.get("traffic_lights", [])
STOP_ZONES     = regions.get("stop_zones", [])
ROIS           = regions.get("rois", [])
BULB_RADIUS    = regions.get("bulb_radius", 8)

num_lanes = max(len(stop_lines), len(traffic_lights), 1)
lanes = []
for i in range(num_lanes):
    lanes.append({
        "traffic_light": traffic_lights[i] if i < len(traffic_lights) else None,
        "stop_line":     stop_lines[i]     if i < len(stop_lines) else None,
        "light_history": deque(maxlen=10),
    })

track_history       = defaultdict(list)
ground_history      = defaultdict(list)
red_violators       = {}
stop_sign_timers    = {}
stop_sign_complied  = set()
stop_sign_violators = set()
near_collision_ids  = set()
ground_history      = defaultdict(list)
ground_last_seen    = {} 
frame_idx = 0  

ground_smoothed = {}
GROUND_SMOOTH_ALPHA = 0.3

STILL_THRESHOLD        = 3
STOP_MIN_SECONDS       = 1
STILL_WINDOW           = 10
STILL_STD_THRESH       = 2.0
STOP_GRACE_FRAMES      = 15
CROSS_LOOKBACK         = 5
PREDICT_FRAMES         = int(fps * 2)
NEAR_COLLISION_DIST    = 15
VELOCITY_SMOOTH_FRAMES = 8
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
BULB_HUE_RANGES = {
    "red":    [(0, 10), (170, 180)],
    "yellow": [(15, 35)],
    "green":  [(40, 90)],
}
BULB_MIN_SATURATION = 60

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
        cv2.putText(frame, label, tuple(quad[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def draw_trail(frame, pts, color):
    for i in range(1, len(pts)):
        p1 = (int(pts[i-1][1]), int(pts[i-1][2]))
        p2 = (int(pts[i][1]), int(pts[i][2]))
        cv2.line(frame, p1, p2, color, 1)

def get_bulb_state(frame, center):
    if center is None:
        return 0.0, 0.0, 0.0
    cx, cy = int(center[0]), int(center[1])
    r  = BULB_RADIUS
    y0 = max(0, cy - r)
    y1 = min(frame.shape[0], cy + r)
    x0 = max(0, cx - r)
    x1 = min(frame.shape[1], cx + r)
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0, 0.0, 0.0
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (cx - x0, cy - y0), r, 255, -1)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    b = cv2.mean(hsv[:, :, 2], mask=mask)[0]
    h = cv2.mean(hsv[:, :, 0], mask=mask)[0]
    s = cv2.mean(hsv[:, :, 1], mask=mask)[0]
    return b, h, s

def hue_valid(mean_h, mean_s, bulb):
    if mean_s < BULB_MIN_SATURATION:
        return False
    return any(lo <= mean_h <= hi for lo, hi in BULB_HUE_RANGES[bulb])

def get_light_state(frame, tl):
    brightness = {}
    hue_ok = {}
    for bulb in ["red", "yellow", "green"]:
        center = tl.get(bulb)
        if center is None:
            continue
        b, mean_h, mean_s = get_bulb_state(frame, center)
        brightness[bulb] = b
        hue_ok[bulb] = b > 10 and hue_valid(mean_h, mean_s, bulb)

    if not brightness:
        return "unknown"

    # primary checking brightest bulb hue
    hue_valid_bulbs = {k: v for k, v in brightness.items() if hue_ok[k]}
    if hue_valid_bulbs:
        return max(hue_valid_bulbs, key=hue_valid_bulbs.get)

    # fallback raw brightness
    best = max(brightness, key=brightness.get)
    return best if brightness[best] > 10 else "unknown"


def smoothed_light_state(frame, lane):
    if lane["traffic_light"] is None:
        return "unknown"
    state = get_light_state(frame, lane["traffic_light"])
    lane["light_history"].append(state)
    return max(set(lane["light_history"]), key=lane["light_history"].count)


def draw_bulb_dots(frame, tl, active_state):
    for bulb, color in BULB_COLORS.items():
        center = tl.get(bulb)
        if center is None:
            continue
        cx, cy = int(center[0]), int(center[1])
        is_active = (bulb == active_state)
        thickness = -1 if is_active else 1
        radius = BULB_RADIUS if is_active else max(BULB_RADIUS - 3, 2)
        cv2.circle(frame, (cx, cy), radius, color, thickness)
        cv2.circle(frame, (cx, cy), radius, (255, 255, 255), 1)

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

def check_crossed_stop_line(pts, p1, p2):
    recent = pts[-CROSS_LOOKBACK:]
    for i in range(1, len(recent)):
        if crossed_stop_line(recent[i-1], recent[i], p1, p2):
            return True
    return False

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
    return entry["stationary_frames"] >= max(1, int(round(fps * STOP_MIN_SECONDS)))

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
        if frame_idx - entry["exit_frame"] < STOP_GRACE_FRAMES:
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
            "enter_frame":       frame_idx,
            "last_frame":        frame_idx,
            "last_pos":          (cx, cy),
            "positions":         [],
            "stationary_frames": 0,
            "is_stationary":     False,
        }

    entry = stop_sign_timers[track_id]
    entry["last_frame"] = frame_idx
    entry["positions"].append((cx, cy))
    entry["is_stationary"] = is_stationary(entry["positions"])
    entry["stationary_frames"] = (entry["stationary_frames"] + 1 if entry["is_stationary"] else 0)
    entry["last_pos"] = (cx, cy)

    if track_id not in stop_sign_complied and has_stopped_in_zone(entry, fps):
        stop_sign_complied.add(track_id)

def to_satellite(px, py):
    pt = np.float32([[[px, py]]])
    out = cv2.perspectiveTransform(pt, H)
    return out[0][0]

def get_velocity(gpts):
    if len(gpts) < 2:
        return np.array([0.0, 0.0])
    n = min(VELOCITY_SMOOTH_FRAMES, len(gpts) - 1)
    recent = np.array(gpts[-n-1:])
    deltas = np.diff(recent, axis=0)
    return deltas.mean(axis=0)

def predict_trajectory(gpts, n_frames):
    if len(gpts) < 2:
        return []
    pos = np.array(gpts[-1], dtype=float)
    vel = get_velocity(gpts)
    return [pos + vel * i for i in range(1, n_frames + 1)]

def check_near_collisions(futures):
    ids = list(futures.keys())
    flagged = set()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            fa = futures[ids[i]]
            fb = futures[ids[j]]
            for pa, pb in zip(fa, fb):
                if np.linalg.norm(np.array(pa) - np.array(pb)) < NEAR_COLLISION_DIST:
                    flagged.add(ids[i])
                    flagged.add(ids[j])
                    break
    return flagged

def draw_satellite_overlay(base, ground_hist, futures, near_col, red_viol, stop_viol):
    img = base.copy()

    for i, sl in enumerate(stop_lines):
        p1s = to_satellite(*sl[0]).astype(int)
        p2s = to_satellite(*sl[1]).astype(int)
        cv2.line(img, tuple(p1s), tuple(p2s), (0, 0, 255), 2)
        cv2.putText(img, f"STOP {i}", tuple(p1s),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    for i, zone in enumerate(STOP_ZONES):
        sat_zone = np.array([to_satellite(*pt) for pt in zone], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [sat_zone], True, (0, 255, 255), 1)

    for tid, gpts in ground_hist.items():
        if not gpts:
            continue

        is_near = tid in near_col
        is_red = tid in red_viol
        is_stop = tid in stop_viol
        is_bad = is_red or is_stop

        if is_near and not is_bad:
            color = (0, 165, 255)
        elif is_bad:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        # draw ground trail
        pts_np = [tuple(np.array(p).astype(int)) for p in gpts]
        for i in range(1, len(pts_np)):
            cv2.line(img, pts_np[i-1], pts_np[i], color, 1)

        # current position dot
        cv2.circle(img, pts_np[-1], 6, color, -1)
        cv2.circle(img, pts_np[-1], 6, (255, 255, 255), 1)
        cv2.putText(img, str(tid), (pts_np[-1][0] + 7, pts_np[-1][1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # predicted trajectory
        if tid in futures and futures[tid]:
            fut_pts = [tuple(np.array(p).astype(int)) for p in futures[tid]]
            for i in range(1, len(fut_pts)):
                cv2.line(img, fut_pts[i-1], fut_pts[i], (0, 165, 255), 1)
            cv2.circle(img, fut_pts[-1], 3, (0, 165, 255), -1)

        # near-collision warning circle
        if is_near:
            cv2.circle(img, pts_np[-1], 18, (0, 165, 255), 2)

    cv2.putText(img, "Green=normal, Orange=near collision, Red=violation", (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return img

cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
cv2.namedWindow("Satellite", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tracking", 1280, 720)
cv2.resizeWindow("Satellite", 960, 960)

# Main
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    stale_ids = [tid for tid, e in stop_sign_timers.items() if frame_idx - e["last_frame"] > fps * 3]
    for tid in stale_ids:
        entry = stop_sign_timers.pop(tid)
        if has_stopped_in_zone(entry, fps):
            stop_sign_complied.add(tid)
        elif tid not in stop_sign_complied:
            stop_sign_violators.add(tid)

    # per-lane light states
    lane_states = [smoothed_light_state(frame, lane) for lane in lanes]

    # draw camera-view lane annotations
    for i, lane in enumerate(lanes):
        state = lane_states[i]
        color = LIGHT_COLORS[state]
        if lane["stop_line"]:
            p1 = tuple(lane["stop_line"][0])
            p2 = tuple(lane["stop_line"][1])
            cv2.line(frame, p1, p2, color, 2)
            cv2.putText(frame, f"L{i} STOP", p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if lane["traffic_light"]:
            tl = lane["traffic_light"]
            draw_quad(frame, tl["quad"], color, f"TL{i}: {state.upper()}")
            draw_bulb_dots(frame, tl, state)

    for i, zone in enumerate(STOP_ZONES):
        draw_quad(frame, zone, (0, 255, 255), f"ZONE {i}")
    for roi in ROIS:
        draw_quad(frame, roi, (200, 200, 200))

    cv2.putText(frame, f"Frame: {frame_idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(frame, f"Red violations: {len(red_violators)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Stop violations: {len(stop_sign_violators)}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.25, iou=0.4, half=True, device=0)
    r = results[0]

    futures = {}

    if r.boxes.id is not None:
        ids = r.boxes.id.cpu().numpy().astype(int)
        boxes = r.boxes.xyxy.cpu().numpy()

        for track_id, box in zip(ids, boxes):
            x1, y1, x2, y2 = box
            cx = float((x1 + x2) / 2.0)
            cy = float(y2)

            if not in_any_roi(cx, cy, ROIS):
                continue

            # update track history
            track_history[track_id].append((frame_idx, cx, cy))
            track_history[track_id] = track_history[track_id][-100:]
            pts = track_history[track_id]

            # project to satellite
            gpt = to_satellite(cx, cy)
            if track_id in ground_smoothed:
                prev = ground_smoothed[track_id]
                smoothed = GROUND_SMOOTH_ALPHA * gpt + (1 - GROUND_SMOOTH_ALPHA) * prev
            else:
                smoothed = gpt # first appearance, no history yet
            ground_smoothed[track_id] = smoothed

            ground_history[track_id].append(smoothed) # store smoothed
            ground_history[track_id] = ground_history[track_id][-200:]
            ground_last_seen[track_id] = frame_idx
            
            # predict trajectory on satellite
            futures[track_id] = predict_trajectory(ground_history[track_id], PREDICT_FRAMES)

            # red light check
            lane_idx = get_vehicle_lane(cx, lanes)
            if lane_idx is not None:
                light_state = lane_states[lane_idx]
                lane = lanes[lane_idx]
                if len(pts) >= 2 and light_state == "red" and lane["stop_line"]:
                    p1 = tuple(lane["stop_line"][0])
                    p2 = tuple(lane["stop_line"][1])
                    if check_crossed_stop_line(pts, p1, p2):
                        if track_id not in red_violators:
                            red_violators[track_id] = frame_idx
                            print(f"[{frame_idx}] RED LIGHT VIOLATION: track {track_id} lane {lane_idx}")
            else:
                light_state = "unknown"

            check_stop_sign(track_id, cx, cy, frame_idx, fps)

            # draw
            is_red_v = track_id in red_violators
            is_stop_v = track_id in stop_sign_violators
            is_bad = is_red_v or is_stop_v
            color = (0, 0, 255) if is_bad else (0, 255, 0)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"ID {track_id}", (int(x1), int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            label_y = int(y2) + 20
            if is_red_v:
                cv2.putText(frame, "RED LIGHT VIOLATION", (int(x1), label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                label_y += 20
            if is_stop_v:
                cv2.putText(frame, "STOP SIGN VIOLATION", (int(x1), label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                label_y += 20
            elif track_id in stop_sign_complied:
                cv2.putText(frame, "STOP OK", (int(x1), label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                label_y += 20
            elif track_id in stop_sign_timers:
                entry = stop_sign_timers[track_id]
                stop_secs = entry["stationary_frames"] / fps if fps > 0 else 0.0
                text = f"STOPPING {stop_secs:.1f}s" if entry["is_stationary"] else f"MOVING stop={stop_secs:.1f}s"
                tcol = (0, 215, 255) if entry["is_stationary"] else (0, 165, 255)
                cv2.putText(frame, text, (int(x1), label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, tcol, 2)

            draw_trail(frame, pts, color)

    active_ids = set(ids) if r.boxes.id is not None else set()

    GROUND_GRACE_FRAMES = fps // 2

    stale_ground_ids = [
        tid for tid in list(ground_history.keys())
        if frame_idx - ground_last_seen.get(tid, 0) > GROUND_GRACE_FRAMES
    ]
    for tid in stale_ground_ids:
        ground_history.pop(tid, None)
        ground_smoothed.pop(tid, None)
        futures.pop(tid, None)
    
    # near-collision detection
    near_collision_ids = check_near_collisions(futures)
    if near_collision_ids:
        cv2.putText(frame, f"NEAR COLLISION: {near_collision_ids}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    # satellite overlay
    sat_view = draw_satellite_overlay(
        sat_img, ground_history, futures,
        near_collision_ids, red_violators, stop_sign_violators
    )

    out_cam.write(frame)
    out_sat.write(sat_view)
    cv2.imshow("Tracking", frame)
    cv2.imshow("Satellite", sat_view)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_idx += 1

cap.release()
out_cam.release()
out_sat.release()
cv2.destroyAllWindows()

print(f"Red light violators: {len(red_violators)}")
for tid, fidx in red_violators.items():
    print(f"  Track {tid} at frame {fidx}")
print(f"Stop sign violators: {len(stop_sign_violators)}")
print(f"Stop sign complied: {len(stop_sign_complied)}")
print(f"Near collision events detected throughout run.")
