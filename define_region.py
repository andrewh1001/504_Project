import cv2
import json
import numpy as np

video_path = "bell3.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

display = frame.copy()

traffic_lights = []
rois           = []
stop_signs     = []
stop_zones     = []
stop_lines     = []

BULB_RADIUS = 1
SNAP_RADIUS = 10

BULB_ORDER = ["red", "yellow", "green"]
BULB_COLORS = {
    "red":    (0, 0, 255),
    "yellow": (0, 215, 255),
    "green":  (0, 255, 0),
}
MODE_COLORS = {
    "traffic_light": (0, 0, 255),
    "roi":           (0, 255, 0),
    "stop_sign":     (255, 0, 0),
    "stop_zone":     (0, 255, 255),
    "stop_line":     (255, 255, 0),
    "tl_bulb":       (255, 255, 255),
}

POLY_MODES = ["traffic_light", "stop_sign", "stop_zone"]
LINE_MODES = ["stop_line"]

mode          = "traffic_light"
poly_points   = []
preview_point = None
drawing_line  = False
line_start    = None
tl_bulb_state = None

def pts_to_np(pts):
    return np.array(pts, dtype=np.int32).reshape((-1, 1, 2))

def draw_quad(img, pts, color, label=None, thickness=2):
    cv2.polylines(img, [pts_to_np(pts)], True, color, thickness)
    if label:
        cv2.putText(img, label, tuple(pts[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

def draw_bulb(img, center, bulb, ghost=False):
    color = BULB_COLORS[bulb]
    cx, cy = int(center[0]), int(center[1])
    if ghost:
        cv2.circle(img, (cx, cy), BULB_RADIUS, color, 1)
    else:
        cv2.circle(img, (cx, cy), BULB_RADIUS, color, -1)
        cv2.circle(img, (cx, cy), BULB_RADIUS, (255, 255, 255), 1)

def near_first_point(x, y):
    if len(poly_points) < 3:
        return False
    dx = x - poly_points[0][0]
    dy = y - poly_points[0][1]
    return (dx*dx + dy*dy) < SNAP_RADIUS * SNAP_RADIUS

def draw_in_progress(img):
    if not poly_points:
        return
    color = MODE_COLORS.get(mode, (255, 255, 255))
    for pt in poly_points:
        cv2.circle(img, tuple(pt), 4, color, -1)
    for a, b in zip(poly_points, poly_points[1:]):
        cv2.line(img, tuple(a), tuple(b), color, 1)
    if preview_point:
        cv2.line(img, tuple(poly_points[-1]), preview_point, color, 1)

    # roi: show closing edge and snap indicator
    if mode == "roi" and len(poly_points) >= 3 and preview_point:
        snapping = near_first_point(preview_point[0], preview_point[1])
        close_color = (0, 255, 255) if snapping else color
        cv2.line(img, tuple(poly_points[-1]), tuple(poly_points[0]), close_color, 1)
        if snapping:
            cv2.circle(img, tuple(poly_points[0]), SNAP_RADIUS + 2, (0, 255, 255), 2)


def redraw():
    global display
    display = frame.copy()

    # traffic lights
    for i, tl in enumerate(traffic_lights):
        draw_quad(display, tl["quad"], MODE_COLORS["traffic_light"], f"TL {i}")
        for bulb in BULB_ORDER:
            if tl.get(bulb):
                draw_bulb(display, tl[bulb], bulb)
                cx, cy = int(tl[bulb][0]), int(tl[bulb][1])
                cv2.putText(display, bulb[0].upper(), (cx + BULB_RADIUS + 2, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, BULB_COLORS[bulb], 1)

    # rois
    for i, pts in enumerate(rois):
        draw_quad(display, pts, MODE_COLORS["roi"], f"ROI {i}")

    # stop signs
    for i, pts in enumerate(stop_signs):
        draw_quad(display, pts, MODE_COLORS["stop_sign"], f"STOP {i}")

    # stop zones
    for i, pts in enumerate(stop_zones):
        draw_quad(display, pts, MODE_COLORS["stop_zone"], f"ZONE {i}")

    # stop lines
    for i, (p1, p2) in enumerate(stop_lines):
        cv2.line(display, tuple(p1), tuple(p2), MODE_COLORS["stop_line"], 2)
        mx = (p1[0] + p2[0]) // 2
        my = (p1[1] + p2[1]) // 2
        cv2.putText(display, f"LINE {i}", (mx, my - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, MODE_COLORS["stop_line"], 2)

    draw_in_progress(display)

    # ghost bulb preview
    if mode == "tl_bulb" and preview_point and tl_bulb_state:
        draw_bulb(display, preview_point, tl_bulb_state["phase"], ghost=True)

    # hud line 1
    if tl_bulb_state:
        phase = tl_bulb_state["phase"]
        hud = f"Mode: tl_bulb | TL {tl_bulb_state['tl_idx']} | Click {phase.upper()} bulb | 0-9=switch TL"
        cv2.putText(display, hud, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, BULB_COLORS[phase], 2)
    elif mode == "roi":
        cv2.putText(display, f"Mode: roi  pts: {len(poly_points)}  (Enter or click first point to close)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, MODE_COLORS["roi"], 2)
    else:
        pts_needed = "2" if mode == "stop_line" else "4"
        cv2.putText(display, f"Mode: {mode}  pts: {len(poly_points)}/{pts_needed}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, MODE_COLORS.get(mode, (255, 255, 255)), 2)

    cv2.putText(display, "t=TL  b=bulbs  r=ROI  p=stop sign  z=zone  l=line", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)
    cv2.putText(display, "RMB=undo point/bulb  u=undo shape  Enter=close ROI  s=save  q=quit", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)

def commit_poly():
    pts = [p[:] for p in poly_points]
    poly_points.clear()
    if mode == "traffic_light":
        traffic_lights.append({
            "quad": pts,
            "red": None,
            "yellow": None,
            "green": None,
        })
    elif mode == "stop_sign":
        stop_signs.append(pts)
    elif mode == "stop_zone":
        stop_zones.append(pts)


def commit_roi():
    if len(poly_points) >= 3:
        rois.append([p[:] for p in poly_points])
        poly_points.clear()


def commit_bulb(x, y):
    tl = traffic_lights[tl_bulb_state["tl_idx"]]
    tl[tl_bulb_state["phase"]] = [x, y]
    cur = BULB_ORDER.index(tl_bulb_state["phase"])
    if cur < len(BULB_ORDER) - 1:
        tl_bulb_state["phase"] = BULB_ORDER[cur + 1]
    else:
        print(f"TL {tl_bulb_state['tl_idx']} bulbs complete. Press 0-9 to annotate another TL or t to exit.")
        tl_bulb_state["phase"] = "red"


def undo_shape():
    if mode == "traffic_light" and traffic_lights:
        traffic_lights.pop()
    elif mode == "roi" and rois:
        rois.pop()
    elif mode == "stop_sign" and stop_signs:
        stop_signs.pop()
    elif mode == "stop_zone" and stop_zones:
        stop_zones.pop()
    elif mode == "stop_line" and stop_lines:
        stop_lines.pop()
    elif mode == "tl_bulb" and tl_bulb_state:
        tl = traffic_lights[tl_bulb_state["tl_idx"]]
        cur = BULB_ORDER.index(tl_bulb_state["phase"])
        if cur > 0:
            prev = BULB_ORDER[cur - 1]
            tl[prev] = None
            tl_bulb_state["phase"] = prev
        elif tl.get("red") is not None:
            tl["red"] = None

def mouse_callback(event, x, y, flags, param):
    global drawing_line, line_start, preview_point

    if mode == "tl_bulb":
        if event == cv2.EVENT_MOUSEMOVE:
            preview_point = (x, y)
            redraw()
        elif event == cv2.EVENT_LBUTTONDOWN:
            commit_bulb(x, y)
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            undo_shape()
            redraw()

    elif mode == "roi":
        if event == cv2.EVENT_MOUSEMOVE:
            preview_point = (x, y)
            redraw()
        elif event == cv2.EVENT_LBUTTONDOWN:
            if near_first_point(x, y):
                commit_roi()
                preview_point = None
            else:
                poly_points.append([x, y])
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if poly_points:
                poly_points.pop()
            redraw()

    elif mode in POLY_MODES:
        if event == cv2.EVENT_MOUSEMOVE:
            preview_point = (x, y)
            redraw()
        elif event == cv2.EVENT_LBUTTONDOWN:
            poly_points.append([x, y])
            if len(poly_points) == 4:
                commit_poly()
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if poly_points:
                poly_points.pop()
            redraw()

    elif mode in LINE_MODES:
        if event == cv2.EVENT_MOUSEMOVE:
            if drawing_line:
                preview_point = (x, y)
                redraw()
                cv2.line(display, tuple(line_start), (x, y), MODE_COLORS["stop_line"], 2)
        elif event == cv2.EVENT_LBUTTONDOWN:
            if not drawing_line:
                drawing_line = True
                line_start = [x, y]
            else:
                stop_lines.append([line_start, [x, y]])
                drawing_line = False
                line_start = None
                preview_point = None
                redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if drawing_line:
                drawing_line = False
                line_start = None
                preview_point = None
                redraw()

cv2.namedWindow("Selector", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Selector", 1280, 720)
cv2.setMouseCallback("Selector", mouse_callback)
redraw()

# Main
while True:
    cv2.imshow("Selector", display.copy())
    key = cv2.waitKey(20) & 0xFF

    if key == ord("t"):
        mode = "traffic_light"
        poly_points.clear()
        drawing_line = False
        tl_bulb_state = None
        redraw()

    elif key == ord("b"):
        if not traffic_lights:
            print("Draw a traffic light quad first (press t).")
        else:
            mode = "tl_bulb"
            poly_points.clear()
            tl_bulb_state = {
                "tl_idx": len(traffic_lights) - 1,
                "phase": "red",
            }
            print(f"Annotating bulbs for TL {tl_bulb_state['tl_idx']}. Click RED bulb.")
            for i, tl in enumerate(traffic_lights):
                status = {b: "OK" if tl.get(b) else "None" for b in BULB_ORDER}
                print(f"  TL {i}: {status}")
            redraw()

    elif key == ord("r"):
        mode = "roi"
        poly_points.clear()
        drawing_line = False
        tl_bulb_state = None
        redraw()

    elif key == ord("p"):
        mode = "stop_sign"
        poly_points.clear()
        drawing_line = False
        tl_bulb_state = None
        redraw()

    elif key == ord("z"):
        mode = "stop_zone"
        poly_points.clear()
        drawing_line = False
        tl_bulb_state = None
        redraw()

    elif key == ord("l"):
        mode = "stop_line"
        poly_points.clear()
        drawing_line = False
        tl_bulb_state = None
        redraw()

    elif key == ord("u"):
        poly_points.clear()
        undo_shape()
        redraw()

    elif key == 13: # close ROI polygon
        if mode == "roi":
            commit_roi()
            preview_point = None
            redraw()

    elif ord("0") <= key <= ord("9"):
        n = key - ord("0")
        if mode == "tl_bulb" and tl_bulb_state and n < len(traffic_lights):
            tl_bulb_state["tl_idx"] = n
            tl_bulb_state["phase"] = "red"
            poly_points.clear()
            print(f"Switched to TL {n}. Click RED bulb.")
            redraw()

    elif key == ord("s"):
        data = {
            "traffic_lights": traffic_lights,
            "rois":           rois,
            "stop_signs":     stop_signs,
            "stop_zones":     stop_zones,
            "stop_lines":     stop_lines,
            "bulb_radius":    BULB_RADIUS,
        }
        with open("selected_regions.json", "w") as f:
            json.dump(data, f, indent=2)
        print("Saved to selected_regions.json")

    elif key == ord("q"):
        break

cv2.destroyAllWindows()
