import cv2
import json
import numpy as np

video = "bell"
VIDEO_PATH = f"files/{video}.mp4"
SATELLITE_PATH = f"files/{video}.png"
OUTPUT_PATH = f"files/homography_{video}.json"
MIN_POINTS = 4

cap = cv2.VideoCapture(VIDEO_PATH)
ret, cam_frame = cap.read()
cap.release()
satellite = cv2.imread(SATELLITE_PATH)

cam_pts = []
sat_pts = []
active = "cam"

COLORS = {
    "cam": (0, 255, 0),
    "sat": (0, 165, 255),
}

def redraw_cam():
    img = cam_frame.copy()
    for i, (x, y) in enumerate(cam_pts):
        cv2.circle(img, (x, y), 6, COLORS["cam"], -1)
        cv2.circle(img, (x, y), 6, (255, 255, 255), 1)
        cv2.putText(img, str(i), (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["cam"], 2)

    status = f"CAM pts: {len(cam_pts)}  |  SAT pts: {len(sat_pts)}"
    if active == "cam":
        status += "  | Click CAM point"
    else:
        status += "  | Switch to SAT window and click"
    cv2.putText(img, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(img, "u=undo last pair  s=save & compute H  q=quit", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)

    pair_count = min(len(cam_pts), len(sat_pts))
    cv2.putText(img, f"Pairs: {pair_count}/{max(len(cam_pts), len(sat_pts))}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    return img

def redraw_sat():
    img = satellite.copy()
    for i, (x, y) in enumerate(sat_pts):
        cv2.circle(img, (x, y), 6, COLORS["sat"], -1)
        cv2.circle(img, (x, y), 6, (255, 255, 255), 1)
        cv2.putText(img, str(i), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["sat"], 2)

    status = f"CAM pts: {len(cam_pts)}  |  SAT pts: {len(sat_pts)}"
    if active == "sat":
        status += "  | Click SAT point"
    else:
        status += "  | Switch to CAM window and click"
    cv2.putText(img, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return img

def refresh():
    cv2.imshow("Camera Frame", redraw_cam())
    cv2.imshow("Satellite",    redraw_sat())

def cam_callback(event, x, y, flags, param):
    global active
    if event == cv2.EVENT_LBUTTONDOWN and active == "cam":
        cam_pts.append((x, y))
        active = "sat"
        print(f"  CAM pt {len(cam_pts)-1}: ({x}, {y}) now click matching SAT point")
        refresh()

def sat_callback(event, x, y, flags, param):
    global active
    if event == cv2.EVENT_LBUTTONDOWN and active == "sat":
        sat_pts.append((x, y))
        active = "cam"
        print(f"  SAT pt {len(sat_pts)-1}: ({x}, {y}) to pair {len(sat_pts)-1} complete")
        refresh()

def undo():
    global active
    if cam_pts and sat_pts:
        cp = cam_pts.pop()
        sp = sat_pts.pop()
        active = "cam"
        print(f"  Undid pair {len(cam_pts)}: cam={cp} sat={sp}")
    elif cam_pts:
        cp = cam_pts.pop()
        active = "cam"
        print(f"  Undid unpaired cam pt: {cp}")
    refresh()


def compute_and_save():
    pair_count = min(len(cam_pts), len(sat_pts))
    if pair_count < MIN_POINTS:
        print(f"Need at least {MIN_POINTS} pairs, only have {pair_count}.")
        return False

    c = np.float32(cam_pts[:pair_count])
    s = np.float32(sat_pts[:pair_count])

    H, mask = cv2.findHomography(c, s, cv2.RANSAC, ransacReprojThreshold=4.0)
    if H is None:
        print("Computation failed")
        return False

    inliers = int(mask.sum())
    print(f"\nHomography computed. Inliers: {inliers}/{pair_count}")

    errors = []
    for i, (cp, sp) in enumerate(zip(c, s)):
        if mask[i]:
            proj = cv2.perspectiveTransform(cp.reshape(1, 1, 2), H)[0][0]
            err = np.linalg.norm(proj - sp)
            errors.append(err)
            print(f"  pt {i}: cam={cp.tolist()} sat={sp.tolist()} err={err:.2f}px")
    print(f"  Mean reprojection error: {np.mean(errors):.2f}px")

    data = {
        "H":        H.tolist(),
        "cam_pts":  [list(p) for p in cam_pts[:pair_count]],
        "sat_pts":  [list(p) for p in sat_pts[:pair_count]],
        "inliers":  inliers,
        "n_pairs":  pair_count,
        "mean_reprojection_error_px": float(np.mean(errors)),
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {OUTPUT_PATH}")

    preview = satellite.copy()
    for i, (cp, sp) in enumerate(zip(c, s)):
        proj = cv2.perspectiveTransform(cp.reshape(1, 1, 2), H)[0][0].astype(int)
        sp_i = tuple(sp.astype(int))
        col = (0, 255, 0) if mask[i] else (0, 0, 255)
        cv2.circle(preview, sp_i, 6, (0, 165, 255), -1)
        cv2.circle(preview, tuple(proj), 4, col, -1)
        cv2.line(preview, sp_i, tuple(proj), col, 1)
        cv2.putText(preview, str(i), (sp_i[0]+8, sp_i[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(preview, "Orange=SAT actual Green=projected CAM Red=outlier", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
    cv2.imshow("Reprojection Check", preview)

    return True

cv2.namedWindow("Camera Frame", cv2.WINDOW_NORMAL)
cv2.namedWindow("Satellite", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera Frame", 960, 540)
cv2.resizeWindow("Satellite", 960, 540)
cv2.setMouseCallback("Camera Frame", cam_callback)
cv2.setMouseCallback("Satellite", sat_callback)

refresh()

# Main
while True:
    key = cv2.waitKey(20) & 0xFF

    if key == ord("u"):
        undo()

    elif key == ord("s"):
        saved = compute_and_save()
        if saved:
            print("\nPress 'q' to quit or keep adding points and press 's' again.")

    elif key == ord("q"):
        break

cv2.destroyAllWindows()