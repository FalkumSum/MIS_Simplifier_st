import os
import re
import math
import streamlit as st
import folium
import streamlit.components.v1 as components
from pyproj import Transformer

st.set_page_config(page_title="MIS Line Simplifier", layout="wide")

# =========================
# Utilities
# =========================
def _bad_num(x):
    return not isinstance(x, (float, int)) or not math.isfinite(x)

def _clean_latlon(seq):
    """Drop invalid/out-of-range points."""
    out = []
    for lat, lon in seq:
        if _bad_num(lat) or _bad_num(lon):
            continue
        if abs(lat) > 90 or abs(lon) > 180:
            continue
        out.append((lat, lon))
    return out

def detect_epsg_from_name(name: str):
    """
    Detect EPSG code in filename/path.
    Matches:
      - 'EPSG:XXXX', 'EPSGXXXX', 'epsg_XXXX', etc. (4–6 digits)
      - bare 326xx / 327xx tokens (UTM north/south)
    Returns int or None.
    """
    if not name:
        return None
    s = name
    m = re.search(r'epsg[:\s_\-]?(\d{4,6})', s, flags=re.IGNORECASE)  # also catches 'EPSG32631'
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    m2 = re.search(r'(326\d{2}|327\d{2})', s, flags=re.IGNORECASE)
    if m2:
        try:
            return int(m2.group(1))
        except ValueError:
            pass
    return None

def detect_epsg_from_text(text: str):
    """Detect EPSG patterns inside file content. Returns int or None."""
    if not text:
        return None
    m = re.search(r'epsg[:\s_\-]?(\d{4,6})', text, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    m2 = re.search(r'(326\d{2}|327\d{2})', text, flags=re.IGNORECASE)
    if m2:
        try:
            return int(m2.group(1))
        except ValueError:
            pass
    return None

def guess_hemisphere_from_northings(original_lines):
    """Heuristic hemisphere guess from UTM northings."""
    Ns = [y for line in original_lines for (_, y) in line]
    if not Ns:
        return "N"
    medN = sorted(Ns)[len(Ns)//2]
    return "S" if medN >= 8_000_000 else "N"

def best_guess_epsg(filename: str, text: str, original_lines):
    """
    Best guess for EPSG:
      1) filename
      2) file content
      3) hemisphere from E/N (fallback placeholder zone)
         - N -> 32631
         - S -> 32740 (Rodrigues)
    """
    by_name = detect_epsg_from_name(filename or "")
    if by_name:
        return by_name
    by_text = detect_epsg_from_text(text or "")
    if by_text:
        return by_text
    hem = guess_hemisphere_from_northings(original_lines)
    return 32740 if hem == "S" else 32631

# =========================
# Parsing
# =========================
def is_coord_line(line: str) -> bool:
    """True if line begins with at least two float-like tokens (x y ...)."""
    if not line:
        return False
    parts = line.strip().split()
    if len(parts) < 2:
        return False
    try:
        float(parts[0]); float(parts[1])
        return True
    except ValueError:
        return False

@st.cache_data(show_spinner=False)
def parse_lines(text: str):
    """
    Parse MIS-like text into 'lines'.
    A new line starts at any non-coordinate header. Blank lines are ignored.
    Returns list of dicts: {header, coord_pairs, coord_texts}
    """
    raw = text.splitlines()
    lines = []
    header = None
    coords, coord_txts = [], []

    def flush():
        nonlocal header, coords, coord_txts
        if coords:
            lines.append({"header": header, "coord_pairs": coords, "coord_texts": coord_txts})
        header = None
        coords, coord_txts = [], []

    for line in raw:
        if line.strip() == "":
            # ignore blanks; they do NOT split lines
            continue

        if is_coord_line(line):
            parts = line.strip().split()
            try:
                x, y = float(parts[0]), float(parts[1])
                coords.append((x, y))
                coord_txts.append(line.strip())
            except ValueError:
                pass
        else:
            # non-coordinate header starts a new line
            if coords:
                flush()
            header = line.strip()

    flush()
    return lines

@st.cache_data(show_spinner=False)
def build_simplified(text: str):
    """
    Create simplified MIS:
      • For each parsed line, keep the header (line name), then the first and last XY row.
    Returns: simplified_text, original_lines, simplified_segments, n_lines, line_names
    """
    parsed = parse_lines(text)
    simplified_text_rows = []
    original_lines = []
    simplified_segments = []
    line_names = []
    visible_idx = 0

    for ln in parsed:
        coords = ln["coord_pairs"]
        texts = ln["coord_texts"]
        hdr = ln["header"]
        if not coords:
            continue

        visible_idx += 1
        original_lines.append(coords)

        name = hdr if hdr else f"LINE {visible_idx}"
        line_names.append(name)

        simplified_text_rows.append(name)
        simplified_text_rows.append(texts[0])
        if len(texts) > 1:
            simplified_text_rows.append(texts[-1])

        simplified_segments.append([coords[0], coords[-1]])  # first & last (or same point)

    simplified_text = "\n".join(simplified_text_rows) + ("\n" if simplified_text_rows else "")
    return simplified_text, original_lines, simplified_segments, len(original_lines), line_names

# =========================
# Lengths (meters, in projected CRS)
# =========================
def _line_length_m(coords):
    """Sum of straight segments in projected XY (meters)."""
    if not coords or len(coords) < 2:
        return 0.0
    total = 0.0
    x0, y0 = coords[0]
    for x1, y1 in coords[1:]:
        total += math.hypot(x1 - x0, y1 - y0)
        x0, y0 = x1, y1
    return total

def compute_lengths(projected_original_lines, projected_simplified_segments):
    """
    Returns two lists (same order as lines):
      - original_lengths_m
      - simplified_lengths_m
    """
    orig = [_line_length_m(line) for line in projected_original_lines]
    simp = [math.hypot(seg[1][0] - seg[0][0], seg[1][1] - seg[0][1]) if len(seg) == 2 else 0.0
            for seg in projected_simplified_segments]
    return orig, simp

# =========================
# Transform & Map
# =========================
@st.cache_data(show_spinner=False)
def transform_to_wgs84(original_lines, simplified_segments, epsg_from: int):
    """Transform projected (x,y) to WGS84 (lat,lon)."""
    if not original_lines:
        return [], []
    transformer = Transformer.from_crs(f"EPSG:{epsg_from}", "EPSG:4326", always_xy=True)

    def tr_pair(x, y):
        lon, lat = transformer.transform(x, y)
        return (lat, lon)

    orig_ll = [_clean_latlon([tr_pair(x, y) for (x, y) in line]) for line in original_lines]
    simp_ll = [_clean_latlon([tr_pair(*seg[0]), tr_pair(*seg[1])]) for seg in simplified_segments]
    return orig_ll, simp_ll

@st.cache_data(show_spinner=False)
def render_map_html(original_lines, simplified_segments, line_names, epsg_from: int, zoom_start: int,
                    show_original: bool, show_simplified: bool) -> str:
    """
    Build Folium map and return static HTML (map interactions don’t rerun the app).
    Tooltips show the line name (header) and lengths.
    """
    if not original_lines:
        return ""

    # Lengths in projected meters (use raw XY)
    orig_lengths_m, simp_lengths_m = compute_lengths(original_lines, simplified_segments)

    # Coordinates in WGS84 for display
    orig_ll, simp_ll = transform_to_wgs84(original_lines, simplified_segments, epsg_from)

    # Map center
    lats = [lat for line in orig_ll for (lat, lon) in line] or [lat for seg in simp_ll for (lat, lon) in seg]
    lons = [lon for line in orig_ll for (lat, lon) in line] or [lon for seg in simp_ll for (lat, lon) in seg]
    if not lats or not lons:
        return ""
    cy = sum(lats) / len(lats)
    cx = sum(lons) / len(lons)

    m = folium.Map(location=[cy, cx], zoom_start=zoom_start, control_scale=True)

    # Original lines layer
    if show_original:
        fg_orig = folium.FeatureGroup(name="Original lines", show=True)
        any_drawn = False
        for i, line in enumerate(orig_ll):
            name = line_names[i] if i < len(line_names) else f"LINE {i+1}"
            tip = folium.Tooltip(
                f"Line: {name}<br>"
                f"Original length: {orig_lengths_m[i]:,.1f} m<br>"
                f"Simplified length: {simp_lengths_m[i]:,.1f} m",
                sticky=True
            )
            if len(line) >= 2:
                folium.PolyLine(line, color="#444444", weight=2, opacity=0.85, tooltip=tip).add_to(fg_orig)
            else:
                folium.CircleMarker(location=line[0], radius=3, color="#444444",
                                    fill=True, fill_opacity=0.9, tooltip=tip).add_to(fg_orig)
            any_drawn = True
        if any_drawn:
            fg_orig.add_to(m)

    # Simplified lines layer
    if show_simplified:
        fg_simp = folium.FeatureGroup(name="Simplified (first–last)", show=True)
        for i, seg in enumerate(simp_ll):
            name = line_names[i] if i < len(line_names) else f"LINE {i+1}"
            tip = folium.Tooltip(
                f"Line: {name}<br>"
                f"Original length: {orig_lengths_m[i]:,.1f} m<br>"
                f"Simplified length: {simp_lengths_m[i]:,.1f} m",
                sticky=True
            )
            if len(seg) == 2 and seg[0] == seg[1]:
                folium.CircleMarker(location=seg[0], radius=5, color="red",
                                    fill=True, fill_opacity=1, tooltip=tip).add_to(fg_simp)
            else:
                folium.PolyLine([seg[0], seg[1]], color="red", weight=4, opacity=0.95,
                                tooltip=tip).add_to(fg_simp)
                # Start/End markers
                folium.CircleMarker(location=seg[0], radius=4, color="red",
                                    fill=True, fill_opacity=1,
                                    tooltip=folium.Tooltip(f"{name} — Start", sticky=True)).add_to(fg_simp)
                folium.CircleMarker(location=seg[1], radius=4, color="red",
                                    fill=True, fill_opacity=1,
                                    tooltip=folium.Tooltip(f"{name} — End", sticky=True)).add_to(fg_simp)
        fg_simp.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m.get_root().render()

# =========================
# Session state
# =========================
if "raw_text" not in st.session_state:
    st.session_state.raw_text = None
if "source_name" not in st.session_state:
    st.session_state.source_name = None
if "parsed" not in st.session_state:
    st.session_state.parsed = None  # (simplified_text, original_lines, simplified_segments, n_lines, line_names)
if "epsg_guess" not in st.session_state:
    st.session_state.epsg_guess = 32631
if "epsg_current" not in st.session_state:
    st.session_state.epsg_current = 32631
if "rendered_map_html" not in st.session_state:
    st.session_state.rendered_map_html = ""

# =========================
# UI
# =========================
st.title("MIS Line Simplifier")
st.caption("For each line: keep the header (line name) and the first & last coordinate. Projected input → WGS84 map with length tooltips.")

# --- 1) Upload & parse ---
with st.expander("1) Upload & parse", expanded=True):
    uploaded = st.file_uploader("Choose a .mis file", type=None, key="uploader")

    parse_btn = st.button("Parse file and generate simplified mis for download", type="primary", use_container_width=True)

    if parse_btn:
        if not uploaded:
            st.error("Please upload a file first.")
        else:
            text = uploaded.read().decode("utf-8", errors="replace")
            with st.spinner("Parsing…"):
                simplified_text, original_lines, simplified_segments, n_lines, line_names = build_simplified(text)

            # Save state
            st.session_state.raw_text = text
            st.session_state.source_name = uploaded.name
            st.session_state.parsed = (simplified_text, original_lines, simplified_segments, n_lines, line_names)
            st.session_state.rendered_map_html = ""  # reset map

            # EPSG best guess
            epsg = best_guess_epsg(uploaded.name or "", text, original_lines)
            st.session_state.epsg_guess = int(epsg)
            st.session_state.epsg_current = int(epsg)

            st.success(f"Parsed. Lines detected: {n_lines}. EPSG guess: {epsg}.")

# --- 2) Map settings & apply ---
with st.expander("2) Map settings", expanded=True):
    with st.form("settings_form", clear_on_submit=False):
        zoom_start = st.slider("Map zoom", 2, 18, 12)

        # Stacked vertically
        show_original = st.checkbox("Show original lines", value=True)
        show_simplified = st.checkbox("Show simplified lines", value=True)

        epsg_from = st.number_input(
            "EPSG (projected X,Y → WGS84)",
            min_value=2000, max_value=999999,
            value=int(st.session_state.epsg_current),
            step=1,
            help="Prefilled from filename/content; change if needed."
        )

        apply_clicked = st.form_submit_button("Show on map", type="primary")

    if apply_clicked:
        if not st.session_state.parsed:
            st.error("No data yet. Upload and parse first.")
        else:
            st.session_state.epsg_current = int(epsg_from)
            simplified_text, original_lines, simplified_segments, _, line_names = st.session_state.parsed
            try:
                html = render_map_html(
                    original_lines,
                    simplified_segments,
                    line_names,
                    epsg_from=int(epsg_from),
                    zoom_start=int(zoom_start),
                    show_original=bool(show_original),
                    show_simplified=bool(show_simplified),
                )
                st.session_state.rendered_map_html = html
                st.success("Map updated.")
            except Exception as e:
                st.error(f"Map failed: {e}")
                st.info("Check that the EPSG matches your projected coordinates (e.g., 32631 for UTM 31N).")

# --- 3) Results ---
if st.session_state.parsed:
    simplified_text, original_lines, simplified_segments, n_lines, line_names = st.session_state.parsed

    st.subheader("Summary")
    st.write(f"- **Lines detected:** {n_lines}")
    st.write(f"- **Input points:** {sum(len(l) for l in original_lines)}")
    if st.session_state.source_name:
        st.write(f"- **Source:** `{st.session_state.source_name}`")
    st.write(f"- **EPSG in use:** {st.session_state.epsg_current}")

    st.subheader("Map")
    if st.session_state.rendered_map_html:
        components.html(st.session_state.rendered_map_html, height=650, scrolling=False)
    else:
        st.info("Clic \"show map\" to visualize the lines.")

    st.subheader("Download simplified file")
    base = os.path.splitext(st.session_state.source_name or "output.mis")[0]
    st.download_button(
        "Download .mis",
        data=simplified_text.encode("utf-8"),
        file_name=f"{base}_simplified.mis",
        mime="text/plain",
    )

    st.subheader("Preview (first 120 lines)")
    preview = "\n".join(simplified_text.splitlines()[:120])
    st.code(preview if preview else "(empty)", language="text")
else:
    st.info("Upload and parse a file to begin.")
