"""
ElastiqA — web interface.

Run locally::

    pip install -e ".[app]"
    python app.py

Deployed on Hugging Face Spaces from this same file.

All processing lives in ``elastiqa.app_logic``, which imports no Gradio and is
covered by the test suite. This file is wiring and layout only.

Design notes
------------
The palette is taken directly from the figures in the repository, so a
spectrogram on screen and a spectrogram in the report are the same object in
the same colours: steel blue for corrected signals, crimson for broken or
naive ones, on the near-black of an instrument display.

Every result carries its measured numbers underneath, in monospace. That is
the project's thesis rendered as an interface — the claim is not "this sounds
better", it is "this measures better, and here is the number".

The three columns are user-resizable. The two fixed tracks are driven by CSS
custom properties on ``#shell``; drag handles inserted after load write those
properties and persist them per browser.
"""

from __future__ import annotations

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Gradio is not installed. Run:  pip install -e \".[app]\""
    ) from exc

from elastiqa import __version__
from elastiqa import app_logic as al
from elastiqa.pitch.f0 import SCALES

SR = 22050

# Gradio 6 moved `theme` and `css` from the Blocks constructor to launch().
GRADIO_MAJOR = int(gr.__version__.split(".")[0])
THEME_ON_LAUNCH = GRADIO_MAJOR >= 6

ABYSS = "#060A11"
HULL = "#0C1420"
RAISED = "#111C2A"
RULE = "#1A2634"
PRUSSIAN = "#17456E"
SIGNAL = "#5AA9E6"
BRASS = "#C99A5B"
PAPER = "#E4EBF3"
MUTED = "#7C8FA6"

CSS = f"""
.gradio-container, .gradio-container .main {{
  background: {ABYSS} !important;
  max-width: 100% !important;
  font-family: "Public Sans", system-ui, sans-serif;
  color: {PAPER};
}}
footer {{ display:none !important; }}

/* Explicit, non-wrapping three-column shell.
   Gradio's Row wraps its Columns once the viewport drops below the sum of
   their min-widths, and Windows at 125% display scaling turns a 1366px screen
   into 1092 CSS pixels -- narrow enough to trigger it on an ordinary laptop.
   A five-track grid (column / handle / column / handle / column) makes the
   centre track consume every remaining pixel instead of letting Gradio place
   unused width between the columns. The shell only stacks where we say.

   The two fixed tracks read their width from custom properties so the drag
   handles have a single place to write. `min-width: 0 !important` on the
   columns is deliberate: Gradio emits its `min_width` argument as an inline
   style, and a stylesheet !important outranks an inline declaration, so the
   dragged value stays authoritative. */
#shell {{
  display: grid !important;
  grid-template-columns:
    var(--rail-w) 7px minmax(0, 1fr) 7px var(--preview-w);
  gap: 0 !important; align-items: stretch; width: 100%;
  --rail-w: 224px; --preview-w: 340px;
}}
#rail, #stage, #preview {{ box-sizing: border-box !important; }}
#rail {{
  grid-column: 1;
  flex: 0 0 var(--rail-w) !important; width: var(--rail-w) !important;
  min-width: 0 !important; max-width: none !important;
  background: {HULL};
  padding: 1.3rem 0.9rem 1.3rem 1.1rem;
  gap: 0 !important;
}}
#stage {{
  grid-column: 3;
  flex: 1 1 0% !important; min-width: 0 !important;
  padding: 1.4rem 1.7rem;
}}
#stage > * {{
  width: 100% !important; max-width: none !important; min-width: 0 !important;
}}
#preview {{
  grid-column: 5;
  flex: 0 0 var(--preview-w) !important; width: var(--preview-w) !important;
  min-width: 0 !important; max-width: none !important;
  background: {HULL}; padding: 1.3rem;
}}

/* The handles stand in for the borders that used to separate the tracks:
   a wide-enough grab target with a hairline drawn down the middle of it. */
.col-resizer {{
  flex: 0 0 7px !important; width: 7px; align-self: stretch;
  position: relative; cursor: col-resize; z-index: 30;
  background: transparent; touch-action: none;
}}
.col-resizer::after {{
  content: ""; position: absolute; top: 0; bottom: 0; left: 3px;
  width: 1px; background: {RULE};
  transition: background 120ms ease, width 120ms ease;
}}
.col-resizer:hover::after, .col-resizer:focus-visible::after,
.col-resizer.dragging::after {{
  background: {SIGNAL}; width: 2px; left: 2.5px;
}}
.col-resizer:focus-visible {{ outline: none; }}

/* The rail holds bare nav items, not cards: strip Gradio's block chrome and
   the inter-block gap that was spreading them a full line apart. */
#rail > div, #rail .form, #rail .block {{
  background: transparent !important; border: none !important;
  box-shadow: none !important; padding: 0 !important;
  margin: 0 !important; gap: 0 !important; min-width: 0 !important;
}}

#wordmark {{ margin-bottom: 1.1rem; }}
#wordmark .name {{
  font-family: "Newsreader", Georgia, serif;
  font-size: 1.65rem; font-weight: 500; letter-spacing: -0.015em;
  color: {PAPER}; line-height: 1;
}}
#wordmark .ver {{
  font-family: "JetBrains Mono", monospace; font-size: 0.66rem;
  color: {MUTED}; letter-spacing: 0.08em; margin-top: 0.35rem;
}}
#harmonics {{ margin: 0.9rem 0 1.1rem 0; opacity: 0.85; }}

#rail button {{
  width: 100%; text-align: left !important;
  justify-content: flex-start !important;
  min-height: 0 !important; height: auto !important;
  background: transparent !important; border: none !important;
  border-left: 2px solid transparent !important; border-radius: 0 !important;
  color: {MUTED} !important; font-size: 0.9rem !important;
  font-weight: 450 !important; padding: 0.5rem 0 0.5rem 0.8rem !important;
  margin: 0 !important; box-shadow: none !important;
  transition: color 120ms ease, border-color 120ms ease;
}}
#rail button:hover {{ color: {PAPER} !important; }}
#rail button.primary {{
  color: {PAPER} !important; border-left: 2px solid {SIGNAL} !important;
  background: linear-gradient(90deg, rgba(23,69,110,0.55), transparent) !important;
}}

.eyebrow {{
  font-family: "JetBrains Mono", monospace; font-size: 0.62rem;
  letter-spacing: 0.16em; text-transform: uppercase; color: {MUTED};
}}
#rail .eyebrow {{ margin: 1.15rem 0 0.4rem 0.8rem; }}
#rail .eyebrow:first-of-type {{ margin-top: 0.5rem; }}
#railfoot {{
  margin-top: 1.5rem !important; padding: 0.85rem; border: 1px solid {RULE};
  border-radius: 4px; background: rgba(23,69,110,0.14);
}}
#railfoot p {{ font-size: 0.76rem; color: {MUTED}; margin: 0; line-height: 1.5; }}

.sectionhead {{ margin-bottom: 1.2rem; }}
.sectionhead h2 {{
  font-family: "Newsreader", Georgia, serif; font-size: 1.5rem;
  font-weight: 500; color: {PAPER}; margin: 0.25rem 0 0.3rem 0;
  letter-spacing: -0.01em;
}}
.sectionhead p {{ color: {MUTED}; font-size: 0.88rem; margin: 0; max-width: 60ch; }}

#stage .block, #preview .block {{
  background: {RAISED} !important; border: 1px solid {RULE} !important;
  border-radius: 4px !important;
}}
#stage label span, #preview label span {{
  color: {MUTED} !important; font-size: 0.78rem !important;
  font-weight: 450 !important;
}}
#stage button.primary {{
  background: {PRUSSIAN} !important; border: 1px solid {SIGNAL}44 !important;
  color: {PAPER} !important; border-radius: 3px !important; font-weight: 500 !important;
}}
#stage button.primary:hover {{ background: #1D5786 !important; }}
#stage button.secondary {{
  background: transparent !important; border: 1px solid {RULE} !important;
  color: {MUTED} !important; border-radius: 3px !important;
}}
#stage button.secondary:hover {{ color: {PAPER} !important; }}

.card {{
  border: 1px solid {RULE}; border-radius: 4px; padding: 0.85rem 0.95rem;
  background: {RAISED}; border-top: 2px solid {RULE};
}}
.card.good {{ border-top-color: {SIGNAL}; }}
.card.bad {{ border-top-color: {BRASS}; }}
.card .tag {{
  font-family: "JetBrains Mono", monospace; font-size: 0.6rem;
  letter-spacing: 0.14em; text-transform: uppercase;
}}
.card.good .tag {{ color: {SIGNAL}; }}
.card.bad .tag {{ color: {BRASS}; }}
.card h4 {{
  font-size: 0.94rem; font-weight: 550;
  margin: 0.3rem 0 0.15rem 0; color: {PAPER};
}}
.card p {{ font-size: 0.79rem; color: {MUTED}; margin: 0; line-height: 1.45; }}

#readout {{
  border-top: 1px solid {SIGNAL};
  background: linear-gradient(180deg, rgba(23,69,110,0.20), transparent 70%);
  padding: 0.85rem 0.95rem; margin-top: 0.4rem;
}}
#readout table {{
  font-family: "JetBrains Mono", monospace; font-size: 0.76rem;
  width: 100%; border-collapse: collapse;
}}
#readout thead {{ display: none; }}
#readout td {{ padding: 0.34rem 0; border-bottom: 1px solid {RULE}; }}
#readout td:first-child {{ color: {MUTED}; }}
#readout td:last-child {{ text-align: right; color: {PAPER}; }}
#readout code {{ background: none !important; color: {SIGNAL} !important; padding: 0; }}
#readout tr:last-child td {{ border-bottom: none; }}

.note {{ color: {MUTED}; font-size: 0.78rem; line-height: 1.5; }}
.note code, .note a {{ color: {SIGNAL}; background: none; text-decoration: none; }}

.prose p, .prose li {{ color: #B9C7D8; font-size: 0.88rem; line-height: 1.65; }}
.prose h3 {{
  font-family: "Newsreader", Georgia, serif; color: {PAPER};
  font-weight: 500; font-size: 1.12rem; margin: 1.5rem 0 0.5rem 0;
}}
.prose strong {{ color: {PAPER}; font-weight: 550; }}

/* ------------------------------------------------------------------------
   Professional finish and motion layer. Everything below is presentation
   only: it does not alter component values, events, audio, or navigation.
   ------------------------------------------------------------------------ */

.gradio-container, .gradio-container .main {{
  background:
    radial-gradient(circle at 18% -10%, rgba(90,169,230,0.10), transparent 34rem),
    radial-gradient(circle at 88% 8%, rgba(201,154,91,0.065), transparent 28rem),
    {ABYSS} !important;
}}

#shell {{
  min-height: 100vh;
  isolation: isolate;
  animation: shellReveal 650ms cubic-bezier(0.22, 1, 0.36, 1) both;
}}

#rail, #preview {{
  background:
    linear-gradient(180deg, rgba(17,28,42,0.96) 0%, rgba(12,20,32,0.99) 72%),
    {HULL} !important;
  box-shadow: 0 0 36px rgba(0,0,0,0.20);
}}

#stage {{
  position: relative;
  background:
    linear-gradient(180deg, rgba(90,169,230,0.025), transparent 18rem),
    transparent;
}}
#stage::before {{
  content: "";
  position: absolute; inset: 0;
  pointer-events: none;
  background: linear-gradient(
    110deg, transparent 20%, rgba(255,255,255,0.012) 50%, transparent 80%
  );
  z-index: 0;
}}
#stage > * {{
  position: relative; z-index: 1;
  animation: sectionEnter 520ms cubic-bezier(0.22, 1, 0.36, 1) both;
}}

.sectionhead-shell {{
  position: relative;
  overflow: hidden;
  border: 1px solid rgba(90,169,230,0.17) !important;
  border-radius: 12px !important;
  background:
    linear-gradient(
      135deg, rgba(23,69,110,0.26),
      rgba(17,28,42,0.90) 52%, rgba(17,28,42,0.72)
    );
  box-shadow:
    0 14px 34px rgba(0,0,0,0.18),
    inset 0 1px 0 rgba(255,255,255,0.025) !important;
}}
.sectionhead {{
  position: relative;
  margin: 0;
  padding: 0.25rem 0.35rem 0.55rem;
}}
.sectionhead::after {{
  content: "";
  position: absolute; left: 1.25rem; bottom: 0; width: 5.5rem; height: 2px;
  border-radius: 999px;
  background: linear-gradient(90deg, {SIGNAL}, transparent);
}}
.sectionhead h2 {{
  font-size: clamp(1.55rem, 2.3vw, 2rem);
  text-shadow: 0 2px 18px rgba(90,169,230,0.08);
}}
.sectionhead p {{ color: #A8BAD0; }}

#wordmark .name {{
  width: fit-content;
  background: linear-gradient(110deg, #F4F8FC 15%, #BFDDF5 52%, #F0D4B0 90%);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  filter: drop-shadow(0 4px 16px rgba(90,169,230,0.10));
}}
#wordmark .ver {{ color: #8CA4BE; }}

#harmonics rect {{
  transform-box: fill-box;
  transform-origin: center bottom;
  animation: harmonicPulse 1.85s ease-in-out infinite;
}}
#harmonics rect:nth-child(2) {{ animation-delay: -1.45s; }}
#harmonics rect:nth-child(3) {{ animation-delay: -1.15s; }}
#harmonics rect:nth-child(4) {{ animation-delay: -0.85s; }}
#harmonics rect:nth-child(5) {{ animation-delay: -0.55s; }}
#harmonics rect:nth-child(6) {{ animation-delay: -0.25s; }}
#harmonics rect:nth-child(7) {{ animation-delay: -1.30s; }}
#harmonics rect:nth-child(8) {{ animation-delay: -0.70s; }}

#rail button {{
  position: relative;
  overflow: hidden;
  border-radius: 0 8px 8px 0 !important;
  transition:
    color 180ms ease, border-color 180ms ease, background 180ms ease,
    transform 180ms cubic-bezier(0.22, 1, 0.36, 1), box-shadow 180ms ease !important;
}}
#rail button:hover {{
  transform: translateX(4px);
  background: linear-gradient(90deg, rgba(90,169,230,0.08), transparent) !important;
}}
#rail button.primary {{
  box-shadow: inset 10px 0 24px -22px {SIGNAL};
}}

#railfoot {{
  border-radius: 10px;
  background: linear-gradient(145deg, rgba(23,69,110,0.19), rgba(17,28,42,0.55));
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.025);
  transition: border-color 180ms ease, transform 180ms ease, box-shadow 180ms ease;
}}
#railfoot:hover {{
  transform: translateY(-2px);
  border-color: rgba(90,169,230,0.35);
  box-shadow: 0 12px 28px rgba(0,0,0,0.16);
}}

#stage .block, #preview .block {{
  border-radius: 11px !important;
  box-shadow: 0 10px 26px rgba(0,0,0,0.13), inset 0 1px 0 rgba(255,255,255,0.018);
  transition:
    border-color 180ms ease, transform 180ms cubic-bezier(0.22, 1, 0.36, 1),
    box-shadow 180ms ease, background-color 180ms ease;
}}
#stage .block:hover, #preview .block:hover {{
  border-color: rgba(90,169,230,0.28) !important;
  box-shadow: 0 14px 34px rgba(0,0,0,0.18), 0 0 0 1px rgba(90,169,230,0.025);
}}

.card {{
  position: relative;
  overflow: hidden;
  min-height: 7.2rem;
  border-radius: 11px;
  background: linear-gradient(145deg, rgba(20,34,51,0.98), rgba(14,24,37,0.98));
  box-shadow: 0 12px 28px rgba(0,0,0,0.15), inset 0 1px 0 rgba(255,255,255,0.025);
  transition:
    transform 220ms cubic-bezier(0.22, 1, 0.36, 1), border-color 220ms ease,
    box-shadow 220ms ease;
}}
#stage .card-shell {{
  overflow: visible !important;
  padding: 0 !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}}
.card::after {{
  content: "";
  position: absolute; inset: 0;
  pointer-events: none;
  background: linear-gradient(
    115deg, transparent 28%, rgba(255,255,255,0.035) 48%, transparent 68%
  );
  transform: translateX(-115%);
  transition: transform 650ms cubic-bezier(0.22, 1, 0.36, 1);
}}
.card:hover {{
  transform: translateY(-4px);
  box-shadow: 0 18px 38px rgba(0,0,0,0.24), 0 0 26px rgba(90,169,230,0.045);
}}
.card:hover::after {{ transform: translateX(115%); }}
.card.good:hover {{ border-color: rgba(90,169,230,0.46); }}
.card.bad:hover {{
  border-color: rgba(201,154,91,0.46);
  box-shadow: 0 18px 38px rgba(0,0,0,0.24), 0 0 26px rgba(201,154,91,0.04);
}}

#stage button.primary, #stage button.secondary {{
  position: relative;
  overflow: hidden;
  min-height: 2.75rem;
  border-radius: 8px !important;
  transition:
    transform 170ms cubic-bezier(0.22, 1, 0.36, 1), box-shadow 170ms ease,
    background-color 170ms ease, border-color 170ms ease !important;
}}
#stage button.primary {{
  background: linear-gradient(135deg, #1D527F, {PRUSSIAN}) !important;
  box-shadow: 0 9px 24px rgba(23,69,110,0.24), inset 0 1px 0 rgba(255,255,255,0.08);
}}
#stage button.primary::after {{
  content: "";
  position: absolute; top: -70%; left: -25%; width: 18%; height: 240%;
  transform: rotate(24deg);
  background: rgba(255,255,255,0.15);
  transition: left 460ms ease;
}}
#stage button.primary:hover {{
  transform: translateY(-2px);
  background: linear-gradient(135deg, #246493, #1A4D78) !important;
  box-shadow: 0 14px 30px rgba(23,69,110,0.32), 0 0 0 1px rgba(90,169,230,0.16);
}}
#stage button.primary:hover::after {{ left: 120%; }}
#stage button.secondary:hover {{
  transform: translateY(-2px);
  border-color: rgba(90,169,230,0.36) !important;
  box-shadow: 0 10px 24px rgba(0,0,0,0.14);
}}
#stage button:active {{ transform: translateY(0) scale(0.985) !important; }}

#stage input, #stage textarea, #stage select,
#preview input, #preview textarea, #preview select {{
  transition:
    border-color 170ms ease, box-shadow 170ms ease,
    background-color 170ms ease;
}}
#stage input:focus, #stage textarea:focus, #stage select:focus,
#preview input:focus, #preview textarea:focus, #preview select:focus {{
  border-color: rgba(90,169,230,0.62) !important;
  box-shadow: 0 0 0 3px rgba(90,169,230,0.10) !important;
  outline: none !important;
}}
#stage input[type="range"] {{ accent-color: {SIGNAL}; }}

#preview > * {{
  animation: previewEnter 520ms cubic-bezier(0.22, 1, 0.36, 1) both;
}}
#preview > *:nth-child(2) {{ animation-delay: 55ms; }}
#preview > *:nth-child(3) {{ animation-delay: 110ms; }}
#preview > *:nth-child(4) {{ animation-delay: 165ms; }}
#preview > *:nth-child(5) {{ animation-delay: 220ms; }}
#preview img {{
  transition: transform 300ms cubic-bezier(0.22, 1, 0.36, 1), filter 300ms ease;
}}
#preview img:hover {{ transform: scale(1.012); filter: saturate(1.05) contrast(1.02); }}

#readout {{
  border-radius: 11px;
  box-shadow: 0 12px 28px rgba(0,0,0,0.14);
}}
#readout tbody tr {{ transition: background-color 150ms ease; }}
#readout tbody tr:hover {{ background: rgba(90,169,230,0.045); }}

.col-resizer {{ transition: background-color 150ms ease; }}
.col-resizer:hover {{ background: rgba(90,169,230,0.035); }}
.col-resizer.dragging::after {{
  animation: dividerGlow 900ms ease-in-out infinite alternate;
  box-shadow: 0 0 14px rgba(90,169,230,0.55);
}}

* {{ scrollbar-width: thin; scrollbar-color: #27415B transparent; }}
*::-webkit-scrollbar {{ width: 9px; height: 9px; }}
*::-webkit-scrollbar-track {{ background: transparent; }}
*::-webkit-scrollbar-thumb {{
  background: linear-gradient(180deg, #2A4B68, #20364C);
  border: 2px solid {ABYSS}; border-radius: 999px;
}}
*::-webkit-scrollbar-thumb:hover {{
  background: linear-gradient(180deg, #376789, #284A66);
}}

@keyframes shellReveal {{
  from {{ opacity: 0; transform: translateY(8px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes sectionEnter {{
  from {{ opacity: 0; transform: translateY(14px) scale(0.994); filter: blur(2px); }}
  to {{ opacity: 1; transform: translateY(0) scale(1); filter: blur(0); }}
}}
@keyframes previewEnter {{
  from {{ opacity: 0; transform: translateX(12px); }}
  to {{ opacity: 1; transform: translateX(0); }}
}}
@keyframes harmonicPulse {{
  0%, 100% {{ transform: scaleY(0.58); opacity: 0.55; }}
  50% {{ transform: scaleY(1); opacity: 1; }}
}}
@keyframes dividerGlow {{
  from {{ opacity: 0.72; }}
  to {{ opacity: 1; }}
}}

@media (prefers-reduced-motion: reduce) {{
  #shell, #stage > *, #preview > *, #harmonics rect,
  .col-resizer.dragging::after {{ animation: none !important; }}
  *, *::before, *::after {{
    scroll-behavior: auto !important;
    transition-duration: 0.01ms !important;
    transition-delay: 0ms !important;
  }}
}}

/* Narrow the preview column first, then stack. The 1240px rule sets the
   variable rather than the width, so an inline value written by a drag
   outranks it; the 1000px rules are !important and always win. */
@media (max-width: 1240px) {{
  #shell {{ --preview-w: 290px; }}
}}
@media (max-width: 1000px) {{
  #shell {{ display: flex !important; flex-wrap: wrap !important; }}
  .col-resizer {{ display: none !important; }}
  #rail {{ flex: 0 0 100% !important; width: 100% !important;
           max-width: 100% !important;
           border-bottom: 1px solid {RULE}; }}
  #rail button {{ width: auto !important; display: inline-flex !important; }}
  #stage, #preview {{ flex: 1 1 100% !important; min-width: 0 !important;
                      width: 100% !important; }}
  #preview {{ border-top: 1px solid {RULE}; }}
}}
"""

THEME = gr.themes.Base(
    primary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Public Sans"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill=ABYSS,
    block_background_fill=RAISED,
    block_border_width="1px",
    block_label_text_color=MUTED,
    body_text_color=PAPER,
    input_background_fill=HULL,
)

# Bar heights follow 1/n -- the amplitude of the nth partial of a sawtooth.
# A divider drawn from the subject rather than a plain hairline.
_BARS = "".join(
    f'<rect x="{i * 9}" y="{22 - round(20 / (i + 1))}" width="3" '
    f'height="{round(20 / (i + 1))}" fill="{SIGNAL}" opacity="{0.9 - i * 0.09:.2f}"/>'
    for i in range(8)
)
HARMONICS_SVG = (
    f'<svg id="harmonics" width="76" height="24" viewBox="0 0 76 24">{_BARS}</svg>'
)

# Runs once on load. Inserts a drag handle between rail/stage and between
# stage/preview, then writes #shell's width variables as they move. The rail
# handle grows its column as the pointer travels right; the preview handle
# grows its column as the pointer travels left, hence the sign argument.
RESIZE_JS = """
() => {
  const shell = document.getElementById('shell');
  if (!shell || shell.dataset.resizable === '1') return;
  const rail = document.getElementById('rail');
  const stage = document.getElementById('stage');
  const preview = document.getElementById('preview');
  if (!rail || !stage || !preview) return;
  shell.dataset.resizable = '1';

  const KEY = 'elastiqa.columns';
  const DEFAULTS = { rail: 224, preview: 340 };
  const MIN = { rail: 150, preview: 240 };
  const MIN_STAGE = 340;

  const read = () => { try { return JSON.parse(localStorage.getItem(KEY)) || {}; }
                       catch (e) { return {}; } };
  const write = (o) => { try { localStorage.setItem(KEY, JSON.stringify(o)); }
                         catch (e) {} };
  const apply = (which, px) => shell.style.setProperty('--' + which + '-w', px + 'px');

  const saved = read();
  Object.keys(DEFAULTS).forEach(k => { if (saved[k]) apply(k, saved[k]); });

  const clamp = (which, px) => {
    const room = shell.getBoundingClientRect().width;
    const other = (which === 'rail' ? preview : rail).getBoundingClientRect().width;
    const max = Math.max(MIN[which], room - other - MIN_STAGE);
    return Math.round(Math.min(Math.max(px, MIN[which]), max));
  };

  const mount = (which, target, sign, before) => {
    const h = document.createElement('div');
    h.className = 'col-resizer';
    h.tabIndex = 0;
    h.setAttribute('role', 'separator');
    h.setAttribute('aria-orientation', 'vertical');
    h.title = 'Drag to resize \\u00b7 double-click to reset';

    let x0 = 0, w0 = 0;
    const move = (e) => apply(which, clamp(which, w0 + sign * (e.clientX - x0)));
    const up = (e) => {
      h.classList.remove('dragging');
      document.body.style.userSelect = '';
      try { h.releasePointerCapture(e.pointerId); } catch (err) {}
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      const s = read();
      s[which] = Math.round(target.getBoundingClientRect().width);
      write(s);
    };

    h.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      x0 = e.clientX;
      w0 = target.getBoundingClientRect().width;
      h.classList.add('dragging');
      document.body.style.userSelect = 'none';
      try { h.setPointerCapture(e.pointerId); } catch (err) {}
      window.addEventListener('pointermove', move);
      window.addEventListener('pointerup', up);
    });

    h.addEventListener('dblclick', () => {
      apply(which, DEFAULTS[which]);
      const s = read(); delete s[which]; write(s);
    });

    h.addEventListener('keydown', (e) => {
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
      e.preventDefault();
      const step = (e.shiftKey ? 48 : 16) * (e.key === 'ArrowRight' ? 1 : -1);
      const w = clamp(which, target.getBoundingClientRect().width + sign * step);
      apply(which, w);
      const s = read(); s[which] = w; write(s);
    });

    shell.insertBefore(h, before);
  };

  mount('rail', rail, 1, stage);
  mount('preview', preview, -1, preview);
}
"""

SECTIONS = ("overview", "stretch", "pitch", "effects", "method")

DEMOS = al.demo_signals(SR)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def _fail(message: str):
    raise gr.Error(message)


def _navigate(target: str):
    """Show one section and mark its rail item active."""
    return [gr.update(visible=(name == target)) for name in SECTIONS] + [
        gr.update(variant="primary" if name == target else "secondary")
        for name in SECTIONS
    ]


def cb_stretch(audio, stretch, method, show_spectrogram):
    try:
        x, sr, note = al.prepare_audio(audio)
    except ValueError as exc:
        _fail(str(exc))

    y, metrics = al.run_time_stretch(x, sr, float(stretch), method)

    image = None
    if show_spectrogram:
        image = al.spectrogram_image({"input": x, method: y}, sr)

    return (
        al.to_gradio_audio(x, sr),
        al.to_gradio_audio(y, sr),
        al.format_metrics(metrics),
        image,
        note,
    )


def cb_pitch(audio, semitones, method, show_spectrogram):
    try:
        x, sr, note = al.prepare_audio(audio)
    except ValueError as exc:
        _fail(str(exc))

    y, metrics = al.run_pitch_shift(x, sr, float(semitones), method)

    image = None
    if show_spectrogram:
        image = al.spectrogram_image({"input": x, method: y}, sr)

    return (
        al.to_gradio_audio(x, sr),
        al.to_gradio_audio(y, sr),
        al.format_metrics(metrics),
        image,
        note,
    )


def cb_compare_pitch(audio, semitones):
    """Side by side: the same shift with and without formant correction."""
    try:
        x, sr, note = al.prepare_audio(audio)
    except ValueError as exc:
        _fail(str(exc))

    plain, _ = al.run_pitch_shift(
        x, sr, float(semitones), "Plain shift (chipmunk / growl)"
    )
    fixed, metrics = al.run_pitch_shift(
        x, sr, float(semitones), "Formant-preserving (natural)"
    )

    image = al.spectrogram_image(
        {"input": x, "plain shift": plain, "formant-preserved": fixed}, sr
    )
    return (
        al.to_gradio_audio(plain, sr),
        al.to_gradio_audio(fixed, sr),
        al.format_metrics(metrics),
        image,
        note,
    )


def cb_effect(audio, effect, scale, strength, third, fifth):
    try:
        x, sr, note = al.prepare_audio(audio)
    except ValueError as exc:
        _fail(str(exc))

    y, metrics = al.run_effect(
        x, sr, effect, scale=scale, strength=float(strength),
        intervals=(float(third), float(fifth)),
    )
    return (
        al.to_gradio_audio(x, sr),
        al.to_gradio_audio(y, sr),
        al.format_metrics(metrics),
        None,
        note,
    )


def cb_toggle_effect_controls(effect):
    is_autotune = effect == "Auto-tune"
    is_harmony = effect == "Harmoniser"
    return (
        gr.update(visible=is_autotune),
        gr.update(visible=is_autotune),
        gr.update(visible=is_harmony),
        gr.update(visible=is_harmony),
    )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _head(title: str, blurb: str) -> None:
    gr.HTML(
        f'<div class="sectionhead"><h2>{title}</h2><p>{blurb}</p></div>',
        elem_classes="sectionhead-shell",
    )


def _card(tag: str, title: str, blurb: str, signal, kind: str) -> None:
    """One clip in a comparison row: a labelled card above its player."""
    with gr.Column():
        gr.HTML(
            f'<div class="card {kind}"><div class="tag">{tag}</div>'
            f"<h4>{title}</h4><p>{blurb}</p></div>",
            elem_classes="card-shell",
        )
        gr.Audio(al.to_gradio_audio(signal, SR), label=None, show_label=False)


def build() -> gr.Blocks:
    block_kwargs: dict = {"title": "ElastiqA"}
    if not THEME_ON_LAUNCH:
        block_kwargs.update(theme=THEME, css=CSS)

    with gr.Blocks(**block_kwargs) as demo:
        with gr.Row(elem_id="shell"):

            # ---------------- navigation rail ----------------
            with gr.Column(scale=0, min_width=224, elem_id="rail"):
                gr.HTML(
                    f'<div id="wordmark"><div class="name">ElastiqA</div>'
                    f'<div class="ver">PHASE VOCODER &middot; v{__version__}</div>'
                    f"</div>{HARMONICS_SVG}"
                )
                gr.HTML('<div class="eyebrow">Listen</div>')
                nav_overview = gr.Button("The problem", variant="primary")
                gr.HTML('<div class="eyebrow">Tools</div>')
                nav_stretch = gr.Button("Stretch time", variant="secondary")
                nav_pitch = gr.Button("Shift pitch", variant="secondary")
                nav_effects = gr.Button("Effects", variant="secondary")
                gr.HTML('<div class="eyebrow">Reference</div>')
                nav_method = gr.Button("How it works", variant="secondary")
                gr.HTML(
                    '<div id="railfoot"><p>Drag the column dividers to resize; '
                    "double-click one to reset. WAV files work as-is. MP3 and "
                    "M4A need ffmpeg on your PATH.</p></div>"
                )

            # ---------------- workspace ----------------
            with gr.Column(scale=1, min_width=360, elem_id="stage"):

                with gr.Column(visible=True) as sec_overview:
                    _head(
                        "Two things that are easy to get wrong",
                        "Every clip below is the same voice. Play them in order "
                        "— the differences are the whole project.",
                    )

                    gr.HTML('<div class="eyebrow">Speed changes pitch</div>')
                    with gr.Row():
                        _card("Source", "Original", "Untouched, 165 Hz.",
                              DEMOS["original"], "good")
                        _card("Resampling", "Slowed to 1.5x",
                              "Longer, but the pitch fell with it.",
                              DEMOS["naive"], "bad")
                        _card("Phase vocoder", "Slowed to 1.5x",
                              "Same length as the middle clip. Pitch unmoved.",
                              DEMOS["vocoder"], "good")

                    gr.HTML('<div class="eyebrow" style="margin-top:1.5rem">'
                            "Holding the pitch is not enough</div>")
                    with gr.Row():
                        _card("Reused phase", "Analysis phase kept",
                              "Right pitch, wrong phase. Listen for the "
                              "metallic wash underneath.",
                              DEMOS["broken"], "bad")
                        _card("Propagated phase", "Phase locked",
                              "Identical magnitudes. Only the phase differs.",
                              DEMOS["vocoder"], "good")

                    gr.HTML('<div class="eyebrow" style="margin-top:1.5rem">'
                            "Moving pitch without moving the speaker</div>")
                    with gr.Row():
                        _card("Plain shift", "Down seven semitones",
                              "The vocal tract shrank too. It sounds like "
                              "someone else.",
                              DEMOS["pitch_plain"], "bad")
                        _card("Formants held", "Down seven semitones",
                              "Same singer, lower note.",
                              DEMOS["pitch_formant"], "good")

                with gr.Column(visible=False) as sec_stretch:
                    _head(
                        "Stretch time",
                        "Change how long the clip lasts. The pitch stays put — "
                        "unless you choose a method that cannot manage it.",
                    )
                    stretch_in = gr.Audio(
                        sources=["upload", "microphone"], type="filepath",
                        label="Input clip",
                    )
                    stretch_amount = gr.Slider(
                        0.5, 2.0, value=1.5, step=0.05,
                        label="Length  ·  1.0 leaves it alone, 2.0 doubles it",
                    )
                    stretch_method = gr.Dropdown(
                        list(al.STRETCH_METHODS), value=list(al.STRETCH_METHODS)[0],
                        label="Method",
                    )
                    stretch_spec = gr.Checkbox(True, label="Draw spectrograms")
                    stretch_go = gr.Button("Stretch", variant="primary")
                    gr.HTML(
                        '<p class="note" style="margin-top:0.9rem">Run the same '
                        "clip twice, once with broken phase and once with "
                        "locking. The consistency figure in the readout is what "
                        "that difference measures.</p>"
                    )

                with gr.Column(visible=False) as sec_pitch:
                    _head(
                        "Shift pitch",
                        "Move the note without changing how long the clip runs. "
                        "Formant preservation keeps the speaker's identity while "
                        "the pitch moves.",
                    )
                    pitch_in = gr.Audio(
                        sources=["upload", "microphone"], type="filepath",
                        label="Input clip",
                    )
                    pitch_amount = gr.Slider(
                        -12, 12, value=-7, step=1,
                        label="Interval  ·  semitones, -12 is an octave down",
                    )
                    pitch_method = gr.Dropdown(
                        list(al.PITCH_METHODS), value=list(al.PITCH_METHODS)[0],
                        label="Method",
                    )
                    pitch_spec = gr.Checkbox(True, label="Draw spectrograms")
                    with gr.Row():
                        pitch_go = gr.Button("Shift", variant="primary")
                        pitch_ab = gr.Button("Compare formants", variant="secondary")
                    gr.HTML(
                        '<p class="note" style="margin-top:0.9rem">Compare '
                        "formants runs the same interval both ways and loads "
                        "them into Before and After.</p>"
                    )

                with gr.Column(visible=False) as sec_effects:
                    _head(
                        "Effects",
                        "Four things the vocoder makes possible. Auto-tune and "
                        "the harmoniser want singing; robot and whisper work on "
                        "anything.",
                    )
                    fx_in = gr.Audio(
                        sources=["upload", "microphone"], type="filepath",
                        label="Input clip",
                    )
                    fx_choice = gr.Radio(
                        list(al.EFFECTS), value="Auto-tune", label="Effect",
                    )
                    fx_scale = gr.Dropdown(
                        sorted(SCALES), value="chromatic", label="Scale",
                    )
                    fx_strength = gr.Slider(
                        0.0, 1.0, value=1.0, step=0.05,
                        label="Correction  ·  1.0 snaps hard to the grid",
                    )
                    fx_third = gr.Slider(
                        -12, 12, value=4, step=1,
                        label="Harmony one  ·  semitones", visible=False,
                    )
                    fx_fifth = gr.Slider(
                        -12, 12, value=7, step=1,
                        label="Harmony two  ·  semitones", visible=False,
                    )
                    fx_go = gr.Button("Apply", variant="primary")

                with gr.Column(visible=False) as sec_method:
                    _head(
                        "How it works",
                        "The mechanism, and what the measurements say about it.",
                    )
                    gr.Markdown(
                        """
Cut the sound into overlapping slices and take the Fourier transform of each.
To stretch time, write the slices back out further apart than they were read in.
The frequencies never changed, so the pitch does not move.

The catch is **phase**. Each slice records not only which frequencies are
present but where each one sits in its cycle. Move the slices apart, reuse those
recorded positions, and neighbouring slices disagree about the waveform where
they overlap. They partially cancel. That is the metallic quality the field
calls *phasiness*.

The fix is to work out how fast each frequency is actually turning and advance
its phase by the right amount for the new spacing. That is the phase vocoder.

### Three refinements

**Phase locking.** One sinusoid spreads across several neighbouring bins that
share a fixed phase relationship. Propagating them independently lets that
relationship drift. Locking each group to its strongest bin keeps the group
moving as one — worth up to a 24x improvement in consistency here.

**Transient reset.** A drum hit is not a slowly varying sinusoid, so
accumulating phase across it smears the attack. Detecting onsets and adopting
the original phase there recovers about 80% of the lost sharpness.

**Formant preservation.** Shifting pitch by resampling moves the vocal tract
resonances too, which is why a sped-up voice sounds smaller rather than higher.
Separating the spectral envelope from the excitation, shifting only the
excitation, then restoring the envelope keeps the speaker intact.

### Nothing here is asserted without a number

Six methods, five stretch factors, six kinds of audio: 180 measurements in
`results/benchmark.csv`. The finding worth reporting is that no method wins
everywhere. WSOLA is better under compression, the phase-locked vocoder is
better under expansion, and the gap widens the harder you push.
                        """,
                        elem_classes="prose",
                    )
                    gr.HTML(
                        '<p class="note" style="margin-top:1.2rem">Source, tests '
                        'and the full benchmark: <a href="https://github.com/'
                        'RaisaZirus/ElastiqA">github.com/RaisaZirus/ElastiqA</a></p>'
                    )

            # ---------------- preview ----------------
            with gr.Column(scale=0, min_width=340, elem_id="preview"):
                gr.HTML('<div class="eyebrow">Preview</div>')
                prev_before = gr.Audio(label="Before", interactive=False)
                prev_after = gr.Audio(label="After", interactive=False)
                gr.HTML('<div class="eyebrow" style="margin-top:0.9rem">Readout</div>')
                prev_readout = gr.Markdown(
                    "| | |\n|---|---|\n| status | `waiting for a clip` |",
                    elem_id="readout",
                )
                prev_note = gr.Markdown(elem_classes="note")
                prev_image = gr.Image(label=None, show_label=False)

        # ---------------- wiring ----------------
        sections = [sec_overview, sec_stretch, sec_pitch, sec_effects, sec_method]
        navs = [nav_overview, nav_stretch, nav_pitch, nav_effects, nav_method]

        for button, name in zip(navs, SECTIONS):
            button.click(
                lambda target=name: _navigate(target),
                inputs=None,
                outputs=sections + navs,
            )

        preview = [prev_before, prev_after, prev_readout, prev_image, prev_note]

        stretch_go.click(
            cb_stretch,
            [stretch_in, stretch_amount, stretch_method, stretch_spec],
            preview,
        )
        pitch_go.click(
            cb_pitch, [pitch_in, pitch_amount, pitch_method, pitch_spec], preview
        )
        pitch_ab.click(cb_compare_pitch, [pitch_in, pitch_amount], preview)
        fx_choice.change(
            cb_toggle_effect_controls, fx_choice,
            [fx_scale, fx_strength, fx_third, fx_fifth],
        )
        fx_go.click(
            cb_effect,
            [fx_in, fx_choice, fx_scale, fx_strength, fx_third, fx_fifth],
            preview,
        )

        demo.load(fn=None, inputs=None, outputs=None, js=RESIZE_JS)

    return demo


def main() -> None:
    demo = build()
    launch_kwargs: dict = {}
    if THEME_ON_LAUNCH:
        launch_kwargs.update(theme=THEME, css=CSS)
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()