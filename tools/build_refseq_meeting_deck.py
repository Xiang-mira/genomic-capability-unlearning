import html
import os
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESKTOP = Path("/home/teacher1/Desktop")
OUT_DIR = ROOT / "outputs"
PPTX_PATH = OUT_DIR / "RefSeq_Target_Unlearning_Report.pptx"
DOCX_PATH = OUT_DIR / "RefSeq_Target_Unlearning_Speaker_Script.docx"
DESKTOP_PPTX = DESKTOP / PPTX_PATH.name
DESKTOP_DOCX = DESKTOP / DOCX_PATH.name

PHASE1_FIG = ROOT / "figures" / "meeting_phase1_refseq_target.png"
PHASE2_FIG = ROOT / "figures" / "meeting_phase2_refseq_sweep.png"

EMU_PER_IN = 914400
SLIDE_W = 12192000
SLIDE_H = 6858000
BG = "F7F1E6"
INK = "1F2933"
MUTED = "5B6570"
BLUE = "2F6FBB"
ORANGE = "D7872D"
GREEN = "2B9C8A"
RED = "B64A3A"
CARD = "FFFDF8"
LINE = "D8CDBD"


def emu(inches: float) -> int:
    return int(inches * EMU_PER_IN)


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def color_fill(color: str) -> str:
    return f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'


def paragraph(text: str, size: int = 22, color: str = INK, bold: bool = False, font: str = "Aptos") -> str:
    b = "<a:b/>" if bold else ""
    return (
        "<a:p>"
        "<a:pPr/>"
        "<a:r>"
        f'<a:rPr lang="en-US" sz="{size * 100}" dirty="0">{b}'
        f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
        f'<a:latin typeface="{font}"/><a:ea typeface="{font}"/><a:cs typeface="{font}"/>'
        "</a:rPr>"
        f"<a:t>{esc(text)}</a:t>"
        "</a:r>"
        "</a:p>"
    )


def textbox(shape_id: int, x: float, y: float, w: float, h: float, lines, size: int = 22,
            color: str = INK, bold_first: bool = False, fill: str | None = None,
            line: str | None = None, radius: bool = False) -> str:
    if isinstance(lines, str):
        lines = [lines]
    prst = "roundRect" if radius else "rect"
    fill_xml = color_fill(fill) if fill else "<a:noFill/>"
    line_xml = f'<a:ln w="9525">{color_fill(line)}</a:ln>' if line else "<a:ln><a:noFill/></a:ln>"
    body = "".join(
        paragraph(text, size=size, color=color, bold=(bold_first and idx == 0))
        for idx, text in enumerate(lines)
    )
    return f"""
    <p:sp>
      <p:nvSpPr>
        <p:cNvPr id="{shape_id}" name="TextBox {shape_id}"/>
        <p:cNvSpPr txBox="1"/>
        <p:nvPr/>
      </p:nvSpPr>
      <p:spPr>
        <a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>
        <a:prstGeom prst="{prst}"><a:avLst/></a:prstGeom>
        {fill_xml}
        {line_xml}
      </p:spPr>
      <p:txBody>
        <a:bodyPr wrap="square" lIns="91440" tIns="73152" rIns="91440" bIns="73152"/>
        <a:lstStyle/>
        {body}
      </p:txBody>
    </p:sp>
    """


def bullet_box(shape_id: int, x: float, y: float, w: float, h: float, bullets, size: int = 21,
               color: str = INK, fill: str | None = None, title: str | None = None) -> str:
    lines = []
    if title:
        lines.append(title)
    lines.extend([f"- {item}" for item in bullets])
    return textbox(shape_id, x, y, w, h, lines, size=size, color=color, bold_first=bool(title),
                   fill=fill, line=LINE if fill else None, radius=bool(fill))


def title(shape_id: int, text: str, subtitle: str | None = None) -> str:
    parts = [paragraph(text, size=32, color=INK, bold=True)]
    if subtitle:
        parts.append(paragraph(subtitle, size=16, color=MUTED))
    return f"""
    <p:sp>
      <p:nvSpPr><p:cNvPr id="{shape_id}" name="Title {shape_id}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
      <p:spPr>
        <a:xfrm><a:off x="{emu(0.55)}" y="{emu(0.28)}"/><a:ext cx="{emu(12.25)}" cy="{emu(0.85)}"/></a:xfrm>
        <a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln>
      </p:spPr>
      <p:txBody><a:bodyPr wrap="square"/><a:lstStyle/>{''.join(parts)}</p:txBody>
    </p:sp>
    """


def footer(shape_id: int, text: str) -> str:
    return textbox(shape_id, 0.55, 7.05, 12.25, 0.24, text, size=9, color="7B7166")


def image_pic(shape_id: int, rel_id: str, x: float, y: float, w: float, h: float, name: str) -> str:
    return f"""
    <p:pic>
      <p:nvPicPr>
        <p:cNvPr id="{shape_id}" name="{esc(name)}"/>
        <p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr>
        <p:nvPr/>
      </p:nvPicPr>
      <p:blipFill>
        <a:blip r:embed="{rel_id}"/>
        <a:stretch><a:fillRect/></a:stretch>
      </p:blipFill>
      <p:spPr>
        <a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>
        <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
      </p:spPr>
    </p:pic>
    """


def slide_xml(shapes: list[str]) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:bg><p:bgPr>{color_fill(BG)}<a:effectLst/></p:bgPr></p:bg>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      {''.join(shapes)}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>
"""


def rels_xml(rels: list[tuple[str, str, str]]) -> str:
    body = "\n".join(
        f'<Relationship Id="{rid}" Type="{typ}" Target="{target}"/>'
        for rid, typ, target in rels
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
{body}
</Relationships>
"""


def chart_bar(shape_id: int, x: float, y: float, w: float, h: float, items: list[tuple[str, float, str]],
              max_value: float, min_value: float = 0.0) -> str:
    shapes = []
    bar_h = h / max(len(items), 1) * 0.58
    gap = h / max(len(items), 1) * 0.42
    for idx, (label, value, color) in enumerate(items):
        yy = y + idx * (bar_h + gap)
        shapes.append(textbox(shape_id + idx * 3, x, yy - 0.02, 2.8, bar_h + 0.04, label, size=13, color=INK))
        frac = max(0.0, min(1.0, (value - min_value) / (max_value - min_value)))
        shapes.append(textbox(shape_id + idx * 3 + 1, x + 3.0, yy, (w - 4.0) * frac, bar_h,
                              "", size=1, fill=color, line=color, radius=True))
        shapes.append(textbox(shape_id + idx * 3 + 2, x + w - 0.9, yy - 0.02, 0.9, bar_h + 0.04,
                              f"{value:.3f}", size=12, color=MUTED))
    return "".join(shapes)


def make_slides() -> list[tuple[str, list[tuple[str, Path]]]]:
    slides: list[tuple[str, list[tuple[str, Path]]]] = []

    slides.append((
        slide_xml([
            textbox(2, 0.72, 1.15, 11.9, 1.15,
                    ["RefSeq Target Validity and Unlearning Diagnostics"],
                    size=38, color=INK, bold_first=True),
            textbox(3, 0.76, 2.48, 7.8, 0.72,
                    "A control-variable check for whether our probe is measuring a real Evo representation or a dataset shortcut.",
                    size=21, color=MUTED),
            textbox(4, 0.8, 4.2, 3.35, 1.25, ["Step 1", "Is the target a real Evo-readable ability?"], size=20,
                    color=INK, bold_first=True, fill=CARD, line=LINE, radius=True),
            textbox(5, 4.95, 4.2, 3.35, 1.25, ["Step 2", "Can the probe measure that ability reliably?"], size=20,
                    color=INK, bold_first=True, fill=CARD, line=LINE, radius=True),
            textbox(6, 9.1, 4.2, 3.35, 1.25, ["Then", "Use unlearning results as causal evidence."], size=20,
                    color=INK, bold_first=True, fill=CARD, line=LINE, radius=True),
            footer(7, "Meeting deck | Evo-1 RefSeq target analysis")
        ]),
        [],
    ))

    slides.append((
        slide_xml([
            title(2, "Why I re-checked the original host-tropism target"),
            bullet_box(3, 0.75, 1.35, 5.9, 4.85, [
                "The previous target asked the model to infer human tropism from viral sequence.",
                "Controlled splits on Virus-Host-Genomes showed weak cross-taxonomy generalization.",
                "That pattern suggests the probe may learn taxonomy identity or close sequence similarity.",
            ], size=20, fill=CARD, title="What raised the concern"),
            bullet_box(4, 6.95, 1.35, 5.55, 4.85, [
                "Evo is a nucleotide prediction model.",
                "It may be stronger on sequence-level patterns than on a high-level phenotype.",
                "Host tropism can depend on receptor biology, immune context, and annotation bias.",
            ], size=20, fill=CARD, title="Why target fit matters"),
            footer(5, "Main point: before judging unlearning, I wanted to check whether the target itself is well matched to Evo.")
        ]),
        [],
    ))

    slides.append((
        slide_xml([
            title(2, "Why I changed the target"),
            textbox(3, 0.8, 1.18, 11.7, 0.7,
                    "I changed the target to test whether the weak causal story came from the method, or from the target itself.",
                    size=21, color=INK, bold_first=True, fill="FFF9EE", line="C9B68F", radius=True),
            textbox(4, 0.92, 2.2, 5.25, 1.7, ["Previous target", "Human host tropism", "A higher-level phenotype with strong shortcut risk"], size=20,
                    color=INK, bold_first=True, fill=CARD, line=LINE, radius=True),
            textbox(5, 7.0, 2.2, 5.25, 1.7, ["Current implemented target", "Coronaviridae vs non-Coronaviridae", "A RefSeq taxonomy-defined sequence proxy"], size=20,
                    color=INK, bold_first=True, fill=CARD, line=LINE, radius=True),
            textbox(6, 0.92, 4.55, 11.3, 1.1,
                    "Important clarification: this is not strict human-tropic vs non-human-tropic classification. I use it as a proxy to ask whether Evo gives a cleaner sequence-level signal.",
                    size=22, color=RED, fill="FFF7F2", line="E0A899", radius=True),
            footer(7, "The target change is a control-variable check, not a final claim that host tropism has been solved.")
        ]),
        [],
    ))

    slides.append((
        slide_xml([
            title(2, "How I built the RefSeq proxy target"),
            bullet_box(3, 0.72, 1.2, 4.05, 5.15, [
                "Read NCBI taxonomy and RefSeq viral assembly metadata.",
                "Use Coronaviridae family taxid as the positive lineage.",
                "Use other viral assemblies as matched negative candidates.",
            ], size=18, fill=CARD, title="Data source"),
            bullet_box(4, 4.98, 1.2, 3.95, 5.15, [
                "Clean sequences to A, C, G, T, and N.",
                "Sample fixed-length windows.",
                "Match negatives by rough sequence length.",
                "Keep labels balanced across splits.",
            ], size=18, fill=CARD, title="Processing"),
            bullet_box(5, 9.12, 1.2, 3.45, 5.15, [
                "Split Coronaviridae records by species/group.",
                "Assign matched negatives by split and length bin.",
                "This helps, but does not remove all taxonomy shortcuts.",
            ], size=18, fill=CARD, title="Controls"),
            footer(6, "The manifest is label-balanced, but the positive and negative sides can still differ in taxonomy composition.")
        ]),
        [],
    ))

    slides.append((
        slide_xml([
            title(2, "Phase 1 workflow: readout first, then localization"),
            textbox(3, 0.85, 1.45, 3.45, 1.2, ["Extract Evo features", "Mean-pooled hidden states from each layer"], size=20,
                    color=INK, bold_first=True, fill=CARD, line=LINE, radius=True),
            textbox(4, 4.95, 1.45, 3.45, 1.2, ["Train probes", "One balanced logistic probe per layer"], size=20,
                    color=INK, bold_first=True, fill=CARD, line=LINE, radius=True),
            textbox(5, 9.05, 1.45, 3.45, 1.2, ["Patch activations", "Find layers that shift target probability"], size=20,
                    color=INK, bold_first=True, fill=CARD, line=LINE, radius=True),
            bullet_box(6, 1.15, 3.35, 10.9, 2.0, [
                "Representation: next_norm hidden states with mask mean pooling.",
                "Metric: validation and test AUROC.",
                "Localization rule: select layers with strong activation-patching effect.",
            ], size=21, fill="FFF9EE", title="Implementation details that matter"),
            footer(7, "Phase 1 asks: where is the target most readable, and which layers look causally relevant under patching?")
        ]),
        [],
    ))

    slides.append((
        slide_xml([
            title(2, "Phase 1 result: the signal is concentrated around L5-L9"),
            image_pic(3, "rId2", 0.55, 1.15, 12.25, 5.55, "Phase 1 figure"),
            footer(4, "Left: layer-wise probe AUROC. Right: activation-patching effect. Selected layers are L5-L9, with L6 as the primary target layer.")
        ]),
        [("rId2", PHASE1_FIG)],
    ))

    slides.append((
        slide_xml([
            title(2, "How I interpret Phase 1"),
            bullet_box(3, 0.78, 1.25, 5.65, 4.8, [
                "The RefSeq proxy target is easy to read from Evo early and middle layers.",
                "Activation patching gives a narrower candidate range than the earlier host-tropism setup.",
                "Layer 6 is the strongest single patching signal.",
            ], size=20, fill=CARD, title="Positive signal"),
            bullet_box(4, 6.85, 1.25, 5.65, 4.8, [
                "Very high probe performance can also indicate shortcut risk.",
                "The probe may rely on taxonomy, k-mer, GC, or RefSeq source patterns.",
                "So Phase 1 is useful, but not enough to prove target validity.",
            ], size=20, fill="FFF7F2", title="Caution"),
            footer(5, "Phase 1 looks cleaner than before, but it does not close the shortcut question.")
        ]),
        [],
    ))

    slides.append((
        slide_xml([
            title(2, "Phase 2 setup: test whether the localized layers are editable"),
            bullet_box(3, 0.75, 1.25, 4.0, 4.95, [
                "label 1 becomes the forget side.",
                "label 0 becomes the retain/control side.",
                "Unlearning uses only the training split.",
                "Evaluation uses held-out validation and test windows.",
            ], size=18, fill=CARD, title="Forget/retain split"),
            bullet_box(4, 5.0, 1.25, 3.7, 4.95, [
                "GD increases forget loss while preserving retain loss.",
                "RMU pushes forget activations away while matching retain activations to a frozen reference.",
            ], size=18, fill=CARD, title="Methods"),
            bullet_box(5, 8.95, 1.25, 3.62, 4.95, [
                "Full: train all layers.",
                "Localized: train L5-L9 only.",
                "Random: train random non-causal layers as a control.",
            ], size=18, fill=CARD, title="Conditions"),
            footer(6, "Phase 2 asks: are L5-L9 only readout layers, or can they serve as clean causal edit points?")
        ]),
        [],
    ))

    slides.append((
        slide_xml([
            title(2, "Phase 2 result: full-model edits reduce the probe most"),
            image_pic(3, "rId2", 0.55, 1.15, 12.25, 5.55, "Phase 2 figure"),
            footer(4, "Left: residual L5-L9 target AUROC after unlearning. Right: forgetting versus retain PPL cost.")
        ]),
        [("rId2", PHASE2_FIG)],
    ))

    slides.append((
        slide_xml([
            title(2, "What Phase 2 means"),
            chart_bar(3, 0.75, 1.35, 6.0, 3.9, [
                ("GD full", 0.547, BLUE),
                ("RMU full", 0.604, GREEN),
                ("GD localized stable", 0.897, ORANGE),
                ("RMU localized", 0.992, GREEN),
                ("Random control", 0.995, "87919C"),
            ], max_value=1.0, min_value=0.45),
            bullet_box(30, 7.15, 1.25, 5.1, 4.95, [
                "Full GD and full RMU remove much more probe signal than localized runs.",
                "Localized GD only becomes stronger when retain quality breaks down.",
                "Localized RMU preserves retain quality, but barely forgets.",
                "Random controls stay close to the original probe signal.",
            ], size=19, fill=CARD, title="Main readout"),
            footer(31, "Lower AUROC is more forgetting. The full methods move the target signal most; localized methods do not yet look like clean selective edits.")
        ]),
        [],
    ))

    slides.append((
        slide_xml([
            title(2, "The main concern: shortcut can still explain the chain"),
            textbox(3, 0.8, 1.32, 11.75, 0.88,
                    "If the Phase 1 probe reads a shortcut, Phase 2 may only prove that full-model training can damage that shortcut.",
                    size=23, color=INK, bold_first=True, fill="FFF7F2", line="E0A899", radius=True),
            bullet_box(4, 0.95, 2.65, 5.4, 3.0, [
                "Coronaviridae versus non-Coronaviridae taxonomy identity.",
                "Family or genus composition differences.",
                "GC, k-mer, or sequence-source artifacts.",
                "Manual split choices in the current manifest.",
            ], size=19, fill=CARD, title="Possible shortcut sources"),
            bullet_box(5, 6.9, 2.65, 5.0, 3.0, [
                "The target is useful as a proxy.",
                "It is not yet a final human-tropism target.",
                "I should validate the target before treating unlearning as causal evidence.",
            ], size=19, fill="FFF9EE", title="Current interpretation"),
            footer(6, "This is the key decision point: target validity should come before a larger sweep.")
        ]),
        [],
    ))

    slides.append((
        slide_xml([
            title(2, "Discussion questions for tonight"),
            bullet_box(3, 0.75, 1.18, 5.75, 5.15, [
                "Should we pause larger unlearning sweeps until target validity is stronger?",
                "Do we first need a target that is better matched to Evo and less shortcut-prone?",
                "Can we trust Phase 2 if the Phase 1 probe may be reading a split artifact?",
            ], size=19, fill="FFF7F2", title="Core decision"),
            bullet_box(4, 6.82, 1.18, 5.65, 5.15, [
                "Could family/genus composition explain the current RefSeq result?",
                "Should we use a public dataset or a standard split if one exists?",
                "If not, should we run family/genus-held-out, homology-held-out, GC/k-mer baselines, and negative-sampling ablations?",
            ], size=19, fill=CARD, title="Concrete validation checks"),
            footer(5, "The goal is to prove that the probe reads a reliable sequence-level signal, not just the way I split this dataset.")
        ]),
        [],
    ))

    slides.append((
        slide_xml([
            textbox(2, 0.78, 0.85, 11.7, 0.78, "Take-home message", size=36, color=INK, bold_first=True),
            textbox(3, 0.95, 1.65, 11.35, 1.55,
                    "Step 1 and Step 2 are not fully closed yet: first prove the target is real and suitable for Evo, then prove the probe measures it.",
                    size=27, color=INK, fill=CARD, line=LINE, radius=True),
            textbox(4, 0.95, 3.65, 11.35, 1.3,
                    "Phase 1 gives a sharper candidate region around L5-L9, but Phase 2 shows that localized edits do not yet have a strong causal interpretation.",
                    size=25, color=INK, fill="FFF9EE", line="C9B68F", radius=True),
            textbox(5, 1.15, 5.45, 10.95, 0.82,
                    "My request: prioritize target and split validation, then return to localized unlearning and benchmark evaluation.",
                    size=23, color=RED, fill="FFF7F2", line="E0A899", radius=True),
            footer(6, "More sweep numbers may not answer the real question unless the probe target is trustworthy.")
        ]),
        [],
    ))

    return slides


def content_types(n_slides: int) -> str:
    overrides = [
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>',
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    overrides.extend(
        f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, n_slides + 1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  {''.join(overrides)}
</Types>
"""


def presentation_xml(n_slides: int) -> str:
    sld_ids = "\n".join(
        f'<p:sldId id="{255 + i}" r:id="rId{i + 1}"/>'
        for i in range(1, n_slides + 1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
  <p:sldIdLst>{sld_ids}</p:sldIdLst>
  <p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
  <p:defaultTextStyle/>
</p:presentation>
"""


def master_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:bg><p:bgPr>{color_fill(BG)}<a:effectLst/></p:bgPr></p:bg><p:spTree>
    <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
    <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
  </p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>
"""


def layout_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
  <p:cSld name="Blank"><p:bg><p:bgPr>{color_fill(BG)}<a:effectLst/></p:bgPr></p:bg><p:spTree>
    <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
    <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
  </p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>
"""


def theme_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Warm Minimal">
  <a:themeElements>
    <a:clrScheme name="Warm Minimal">
      <a:dk1><a:srgbClr val="1F2933"/></a:dk1><a:lt1><a:srgbClr val="F7F1E6"/></a:lt1>
      <a:dk2><a:srgbClr val="5B6570"/></a:dk2><a:lt2><a:srgbClr val="FFFDF8"/></a:lt2>
      <a:accent1><a:srgbClr val="2F6FBB"/></a:accent1><a:accent2><a:srgbClr val="D7872D"/></a:accent2>
      <a:accent3><a:srgbClr val="2B9C8A"/></a:accent3><a:accent4><a:srgbClr val="B64A3A"/></a:accent4>
      <a:accent5><a:srgbClr val="6F4FA3"/></a:accent5><a:accent6><a:srgbClr val="87919C"/></a:accent6>
      <a:hlink><a:srgbClr val="2F6FBB"/></a:hlink><a:folHlink><a:srgbClr val="6F4FA3"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="Aptos"><a:majorFont><a:latin typeface="Aptos Display"/></a:majorFont><a:minorFont><a:latin typeface="Aptos"/></a:minorFont></a:fontScheme>
    <a:fmtScheme name="Warm Minimal"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>
  </a:themeElements>
</a:theme>
"""


def doc_props() -> tuple[str, str]:
    core = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
                   xmlns:dc="http://purl.org/dc/elements/1.1/"
                   xmlns:dcterms="http://purl.org/dc/terms/"
                   xmlns:dcmitype="http://purl.org/dc/dcmitype/"
                   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>RefSeq Target Unlearning Report</dc:title>
  <dc:creator>Codex</dc:creator>
</cp:coreProperties>
"""
    app = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
            xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application>
</Properties>
"""
    return core, app


def build_pptx() -> None:
    slides = make_slides()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(PPTX_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types(len(slides)))
        z.writestr("_rels/.rels", rels_xml([
            ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument", "ppt/presentation.xml"),
            ("rId2", "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties", "docProps/core.xml"),
            ("rId3", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties", "docProps/app.xml"),
        ]))
        core, app = doc_props()
        z.writestr("docProps/core.xml", core)
        z.writestr("docProps/app.xml", app)
        z.writestr("ppt/presentation.xml", presentation_xml(len(slides)))
        pres_rels = [("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster", "slideMasters/slideMaster1.xml")]
        pres_rels.extend(
            (f"rId{i + 1}", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide", f"slides/slide{i}.xml")
            for i in range(1, len(slides) + 1)
        )
        z.writestr("ppt/_rels/presentation.xml.rels", rels_xml(pres_rels))
        z.writestr("ppt/slideMasters/slideMaster1.xml", master_xml())
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", rels_xml([
            ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout", "../slideLayouts/slideLayout1.xml"),
            ("rId2", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme", "../theme/theme1.xml"),
        ]))
        z.writestr("ppt/slideLayouts/slideLayout1.xml", layout_xml())
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", rels_xml([
            ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster", "../slideMasters/slideMaster1.xml"),
        ]))
        z.writestr("ppt/theme/theme1.xml", theme_xml())

        image_counter = 1
        for idx, (xml, images) in enumerate(slides, start=1):
            z.writestr(f"ppt/slides/slide{idx}.xml", xml)
            rels = [("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout", "../slideLayouts/slideLayout1.xml")]
            for rid, path in images:
                media_name = f"image{image_counter}.png"
                z.write(path, f"ppt/media/{media_name}")
                rels.append((rid, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image", f"../media/{media_name}"))
                image_counter += 1
            z.writestr(f"ppt/slides/_rels/slide{idx}.xml.rels", rels_xml(rels))


SPEAKER_SCRIPT = [
    ("Slide 1 - Core question",
     "This week I focused on a basic question in our setup. Are we measuring a real capability inside Evo, or is the probe mostly picking up a shortcut in the data? I think we need to separate this into two steps. First, is the target itself something Evo should be able to represent? Second, can our probe measure that target in a reliable way? Only after those two steps are solid do the unlearning results become strong evidence."),
    ("Slide 2 - Why I re-checked host tropism",
     "The original target was host tropism: given a viral sequence, predict whether it infects humans. That is a meaningful biological question, but the controlled split results made me worried. When I tested cross-taxonomy generalization, the performance dropped. To me, that suggests the model may not be learning a clean host-tropism signal. It may be using taxonomy, family, genus, or close sequence similarity instead."),
    ("Slide 3 - Target clarification",
     "Because of that, I changed the target as a control check. I wanted to know whether the weak causal story came from the unlearning method, or from the target itself. The previous target was human host tropism, which is a higher-level phenotype and has strong shortcut risk. The current RefSeq implementation is different. It is not strict human-tropic versus non-human-tropic classification. It is Coronaviridae versus non-Coronaviridae viral sequences. I am using it as a sequence-level proxy to see whether Evo gives a cleaner and more focused signal."),
    ("Slide 4 - Dataset construction",
     "For the new target, I built the dataset from NCBI RefSeq viral assemblies and NCBI taxonomy. The positive side comes from Coronaviridae lineage, and the negative side comes from other viral assemblies. I cleaned the sequences and cut them into fixed-length windows. The target-family records are split by species or group, and then matched negatives are assigned by split and length bin. So the dataset is label-balanced, but it is not the same as saying every possible shortcut is controlled. Family and genus composition can still matter."),
    ("Slide 5 - Phase 1 workflow",
     "In Phase 1, I extracted Evo representations layer by layer. I used the normalized hidden representation and mean-pooled across valid tokens. Then I trained one simple linear probe for each layer. After that, I used activation patching to ask a more causal question: if I patch a layer, does the target probability move? This gives us both a readout view and a localization view."),
    ("Slide 6 - Phase 1 figure",
     "This figure shows the Phase 1 result. On the left, the target is very easy to read from early and middle layers. On the right, activation patching points to a narrower region. The selected region is layers five through nine, with layer six as the strongest layer. Compared with the earlier host-tropism setup, this looks more focused. But the very strong probe result is also a warning sign, because it could still come from taxonomy or sequence statistics."),
    ("Slide 7 - Phase 1 interpretation",
     "My interpretation is that the new proxy target gives us a cleaner signal than before, but it does not prove that the target is free of shortcuts. The probe can clearly separate the two sides, and patching gives a focused layer range. But that separation might still be driven by taxonomy identity, k-mer patterns, GC content, or RefSeq-specific artifacts. So I see Phase 1 as useful evidence, but not final evidence."),
    ("Slide 8 - Phase 2 setup",
     "For Phase 2, I used the same manifest to build forget and retain sets. The positive side became the forget set, and the negative side became the retain set. I tested two methods: gradient difference and RMU. I also compared three conditions: full model editing, localized editing on the Phase 1 layers, and random-layer controls. The goal was to see whether the localized layers are only good readout layers, or whether they are actually clean places to edit the target."),
    ("Slide 9 - Phase 2 figure",
     "This figure is the main Phase 2 result. The full-model runs reduce the probe signal the most. The localized runs are much weaker under stable settings. If I train localized GD longer, the probe signal goes down more, but retain quality breaks badly. RMU is the opposite: it keeps retain quality stable, but it barely forgets. The random controls stay close to the original signal, which is a useful sanity check."),
    ("Slide 10 - Phase 2 meaning",
     "The main message is that full-model editing works much better than localized editing. That does not mean the Phase 1 layers are useless. It means they may be better described as readout layers than clean causal edit points. Right now, I do not think the localized result is strong enough to claim selective unlearning of the target."),
    ("Slide 11 - Main concern",
     "My biggest concern is still shortcut risk. Since the current target is Coronaviridae versus other viruses, the probe may be reading taxonomy identity or sequence composition. If that is true, then Phase 2 is not proving that we can forget a biological capability. It is mostly proving that full-model training can damage the representation used by the probe. That is useful, but it is a different claim."),
    ("Slide 12 - Next steps",
     "This is the part I most want to discuss tonight. I think we may need to pause larger unlearning sweeps and first make target validity stronger. My question is: should we first find a target that is truly suitable for Evo and less shortcut-prone, and only then continue with unlearning? At minimum, we need to show that the probe is not just reading a split artifact. The current RefSeq dataset is label-balanced, and I did use species-level splitting and length matching, but family and genus composition could still explain the result. So I want to ask whether we should use a public dataset with a standard split, if one exists. If not, I think we should run stronger controlled checks, like family or genus held-out splits, homology held-out splits, GC and k-mer baselines, and negative-sampling ablations."),
    ("Slide 13 - Take-home message",
     "The take-home message is this: Step one and Step two are not fully closed yet. Phase one gives a sharper candidate region, with layer six as the strongest point. Phase two shows that full-model unlearning can reduce the signal, but localized editing still does not have a strong causal interpretation. If the probe target is wrong, then the later unlearning conclusions are not trustworthy. So I would like to prioritize target and split validation first, and then return to localized unlearning and benchmark evaluation."),
]


def docx_content_types() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""


def w_p(text: str, style: str | None = None, bold: bool = False) -> str:
    style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    bold_xml = "<w:b/>" if bold else ""
    return (
        f"<w:p>{style_xml}<w:r><w:rPr>{bold_xml}</w:rPr>"
        f"<w:t xml:space=\"preserve\">{esc(text)}</w:t></w:r></w:p>"
    )


def build_docx() -> None:
    body = [
        w_p("RefSeq Target Unlearning Speaker Script", "Title", bold=True),
        w_p("First-person script written in clear American English.", None),
    ]
    for heading, script in SPEAKER_SCRIPT:
        body.append(w_p(heading, "Heading1", bold=True))
        body.append(w_p(script, None))
    document = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {''.join(body)}
    <w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1080" w:right="1080" w:bottom="1080" w:left="1080"/></w:sectPr>
  </w:body>
</w:document>
"""
    styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:rFonts w:ascii="Aptos" w:hAnsi="Aptos"/><w:sz w:val="24"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:rPr><w:rFonts w:ascii="Aptos Display" w:hAnsi="Aptos Display"/><w:b/><w:sz w:val="36"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="28"/><w:color w:val="1F2933"/></w:rPr></w:style>
</w:styles>
"""
    with zipfile.ZipFile(DOCX_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", docx_content_types())
        z.writestr("_rels/.rels", rels_xml([
            ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument", "word/document.xml"),
            ("rId2", "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties", "docProps/core.xml"),
            ("rId3", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties", "docProps/app.xml"),
        ]))
        z.writestr("word/document.xml", document)
        z.writestr("word/styles.xml", styles)
        z.writestr("word/_rels/document.xml.rels", rels_xml([
            ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles", "styles.xml"),
        ]))
        core, app = doc_props()
        z.writestr("docProps/core.xml", core)
        z.writestr("docProps/app.xml", app)


def main() -> None:
    if not PHASE1_FIG.exists() or not PHASE2_FIG.exists():
        raise FileNotFoundError("Expected Phase 1 and Phase 2 figures are missing.")
    build_pptx()
    build_docx()
    shutil.copy2(PPTX_PATH, DESKTOP_PPTX)
    shutil.copy2(DOCX_PATH, DESKTOP_DOCX)
    print(f"Wrote {PPTX_PATH}")
    print(f"Wrote {DOCX_PATH}")
    print(f"Copied {DESKTOP_PPTX}")
    print(f"Copied {DESKTOP_DOCX}")


if __name__ == "__main__":
    main()
