"""Mubu MCP — document ↔ Markdown / OPML / FreeMind conversion.

Ported from mubu-integration (liuboacean/mubu-integration) and adapted
for the MCP server context.
"""

from __future__ import annotations

import random
import re
import struct
import time
import xml.etree.ElementTree as ET
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Mubu node helpers (real server schema)
# ---------------------------------------------------------------------------

_NODE_ID_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_NODE_ID_LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def gen_node_id() -> str:
    """Generate a short random node id (10 chars, first char a letter),
    matching the ids observed in real Mubu documents (e.g. VPI4liKaE5)."""
    return random.choice(_NODE_ID_LETTERS) + "".join(
        random.choice(_NODE_ID_CHARS) for _ in range(9)
    )


def gen_member_id() -> str:
    """Generate a random 16-digit member id, like the web client does per
    editing session (server records it in colla/members_v2)."""
    return str(random.randint(10**15, 10**16 - 1))


def _esc_html(s: Any) -> str:
    return str(s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _mubu_node(text: Any, children: List[Dict], checked: bool = False) -> Dict[str, Any]:
    """Build a node in the real Mubu definition schema."""
    return {
        "id": gen_node_id(),
        "text": f"<span>{_esc_html(text)}</span>",
        "modified": int(time.time() * 1000),
        "children": children,
        "collapsed": False,
        "finish": bool(checked),
        "color": "",
        "heading": 0,
        "taskStatus": 0,
    }


# ---------------------------------------------------------------------------
# Image support (markdown ![alt](uri) -> node images field)
# ---------------------------------------------------------------------------

_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")


def parse_image_size(data: bytes) -> Optional[Tuple[int, int]]:
    """Parse image dimensions from raw bytes (PNG/JPEG/GIF/WebP/BMP)."""
    if not data:
        return None
    # PNG
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        w, h = struct.unpack(">II", data[16:24])
        if w > 0 and h > 0:
            return w, h
    # JPEG
    if data[:2] == b"\xff\xd8":
        i = 2
        while i < len(data) - 9:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            seg_len = struct.unpack(">H", data[i + 2:i + 4])[0]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF) and seg_len >= 7:
                h, w = struct.unpack(">HH", data[i + 5:i + 9])
                if w > 0 and h > 0:
                    return w, h
            i += 2 + seg_len
    # GIF
    if data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
        w, h = struct.unpack("<HH", data[6:10])
        if w > 0 and h > 0:
            return w, h
    # WebP
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP" and len(data) >= 30:
        fmt = data[12:16]
        if fmt == b"VP8 " and len(data) >= 30:
            w = struct.unpack("<H", data[26:28])[0] & 0x3FFF
            h = struct.unpack("<H", data[28:30])[0] & 0x3FFF
            if w > 0 and h > 0:
                return w, h
        elif fmt == b"VP8L" and len(data) >= 25:
            b = data[21:25]
            packed = int.from_bytes(b, "little")
            w = (packed & 0x3FFF) + 1
            h = ((packed >> 14) & 0x3FFF) + 1
            if w > 0 and h > 0:
                return w, h
        elif fmt == b"VP8X" and len(data) >= 30:
            w = 1 + int.from_bytes(data[24:27], "little")
            h = 1 + int.from_bytes(data[27:30], "little")
            if w > 0 and h > 0:
                return w, h
    # BMP
    if data[:2] == b"BM" and len(data) >= 26:
        w = struct.unpack("<i", data[18:22])[0]
        h = abs(struct.unpack("<i", data[22:26])[0])
        if w > 0 and h > 0:
            return w, h
    return None


def _mubu_node_with_images(
    text: Any,
    children: List[Dict],
    checked: bool = False,
    images: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """Build a node, optionally attaching an ``images`` list."""
    node = _mubu_node(text, children, checked=checked)
    if images:
        node["images"] = images
    return node


def _strip_markdown_images(line: str) -> str:
    """Remove ``![alt](uri)`` fragments from a text line."""
    return _IMAGE_RE.sub("", line).strip()


def _extract_image_refs(line: str) -> List[Dict[str, str]]:
    """Extract ``![alt](uri)`` references from a line."""
    return [{"alt": m.group(1).strip(), "uri": m.group(2).strip()} for m in _IMAGE_RE.finditer(line)]


def _resolve_image_ref(
    ref: Dict[str, str],
    resolve_image: Optional[Callable[[str, str], Optional[Dict[str, Any]]]],
) -> Optional[Dict[str, Any]]:
    """Resolve an image ref into a Mubu images[] item: {id, uri, ow, oh, w}.

    ``resolve_image(uri, alt)`` is a client-provided callback that returns
    ``{"uri": ..., "ow": ..., "oh": ..., "w": ...}`` or None.
    """
    uri = ref.get("uri") or ""
    if not uri:
        return None
    if resolve_image is not None:
        try:
            resolved = resolve_image(uri, ref.get("alt") or "")
            if resolved:
                return {
                    "id": gen_node_id(),
                    "uri": resolved.get("uri") or uri,
                    "ow": int(resolved.get("ow") or 0),
                    "oh": int(resolved.get("oh") or 0),
                    "w": int(resolved.get("w") or 0),
                }
        except Exception:
            return None
    # 无 resolver（或失败）：http(s) 直接引用原 URL，尺寸未知用 0
    if uri.startswith(("http://", "https://")):
        return {"id": gen_node_id(), "uri": uri, "ow": 0, "oh": 0, "w": 0}
    return None


def markdown_to_mubu_tree(
    md: str,
    resolve_image: Optional[Callable[[str, str], Optional[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """Parse Markdown into a single root node in the real Mubu node schema.

    Returns the root node dict (with nested ``children``), ready to be sent
    inside a ``create`` changeset event. Heading / list / checkbox / note
    structure comes from :func:`markdown_to_doc`.

    ``![alt](uri)`` image syntax is turned into the node ``images`` field:
    - ``uri`` starting with http(s):// is referenced directly;
    - local file paths are uploaded through ``resolve_image(uri, alt)``
      (a client callback returning ``{"uri", "ow", "oh", "w"}``);
    - without a resolver, http(s) URIs are kept and local paths are left
      as literal text.
    """
    parsed = markdown_to_doc(md)
    root_in = parsed.get("node") or {}
    text = root_in.get("text") or ""

    def conv(node: Dict[str, Any]) -> Dict[str, Any]:
        node_text = node.get("text") or ""
        images = None
        if _IMAGE_RE.search(node_text):
            refs = _extract_image_refs(node_text)
            images = []
            kept = node_text
            for ref in refs:
                item = _resolve_image_ref(ref, resolve_image)
                if item:
                    images.append(item)
                    # 移除已成功解析的标记
                    kept = _IMAGE_RE.sub(
                        lambda m: "" if m.group(2).strip() == ref.get("uri") else m.group(0),
                        kept,
                        count=1,
                    )
            kept = kept.strip()
            if not images:
                images = None
            else:
                node_text = kept
        return _mubu_node_with_images(
            node_text,
            [conv(c) for c in node.get("children") or []],
            checked=bool(node.get("checked") or node.get("finish")),
            images=images,
        )

    root = conv(root_in)
    if not text and not root["children"]:
        root["text"] = "<span>未命名文档</span>"
    return root


# ---------------------------------------------------------------------------
# Markdown export
# ---------------------------------------------------------------------------


def doc_to_markdown(node: Dict[str, Any], level: int = 0) -> str:
    """Recursively render a node subtree as a Markdown list fragment."""
    lines: List[str] = []
    text = (node.get("text") or "").replace("\n", " ")
    indent = " " * (2 * level)

    if node.get("checked") is not None or node.get("finish") is not None:
        mark = "x" if (node.get("checked") or node.get("finish")) else " "
        lines.append(f"{indent}- [{mark}] {text}")
    else:
        lines.append(f"{indent}- {text}")

    for img in node.get("images") or []:
        uri = img.get("uri") or ""
        if uri:
            lines.append(f"{indent}  ![]( {uri} )")

    for child in node.get("children") or []:
        lines.append(doc_to_markdown(child, level + 1))

    note = node.get("note")
    if note:
        lines.append(f"{indent}> {note}")

    return "\n".join(lines)


def export_markdown(doc: Dict[str, Any]) -> str:
    """Convert a Mubu document structure to Markdown text.

    Handles both ``{"nodes": [...]}`` (API shape) and
    ``{"node": {...}}`` (round-trip shape).
    """
    nodes = doc.get("nodes")
    if nodes:
        lines: List[str] = []
        for node in nodes:
            title = (node.get("text") or "").replace("\n", " ")
            lines.append(f"# {title}")
            for child in node.get("children") or []:
                lines.append(doc_to_markdown(child, level=0))
            note = node.get("note")
            if note:
                lines.append(f"> {note}")
        return "\n".join(lines)

    root = doc.get("node") or doc
    if not isinstance(root, dict) or (not root.get("text") and not root.get("children")):
        raise ValueError("Invalid document structure")
    title = (root.get("text") or "").replace("\n", " ")
    lines = [f"# {title}"]
    for child in root.get("children") or []:
        lines.append(doc_to_markdown(child, level=0))
    note = root.get("note")
    if note:
        lines.append(f"> {note}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown import
# ---------------------------------------------------------------------------


def markdown_to_doc(md: str) -> Dict[str, Any]:
    """Parse Markdown text into a Mubu document structure.

    Returns ``{"node": {"id": "root", "text": ..., "children": [...]}}``.
    """
    root_node: Dict[str, Any] = {"id": "root", "text": "", "children": []}
    counter = [0]

    def next_id() -> str:
        counter[0] += 1
        return f"node_{counter[0]}"

    lines = md.split("\n")
    headings: List[str] = []
    for line in lines:
        if not line.strip():
            continue
        m = re.match(r"^#+\s+(.*)$", line)
        if m:
            headings.append(m.group(1).strip())

    has_list = any(re.match(r"^\s*-\s+", line) for line in lines if line.strip())

    if not headings and not has_list:
        text = md.strip()
        if text:
            root_node["text"] = text
        return {"node": root_node}

    if headings:
        root_node["text"] = headings[0]
        for h in headings[1:]:
            root_node["children"].append({"id": next_id(), "text": h, "children": []})

    if not has_list:
        return {"node": root_node}

    stack: List[Any] = [(0, root_node)]

    for line in lines:
        if not line.strip():
            continue
        if re.match(r"^#+\s+", line):
            continue

        m_note = re.match(r"^(\s*)>\s+(.*)$", line)
        if m_note:
            note_depth = len(m_note.group(1)) // 2
            note_text = m_note.group(2).strip()
            target = root_node
            for d, n in reversed(stack):
                if d == note_depth:
                    target = n
                    break
            target["note"] = note_text
            continue

        m_list = re.match(r"^(\s*)-\s+(.*)$", line)
        if m_list:
            indent = len(m_list.group(1))
            depth = indent // 2
            content = m_list.group(2).strip()

            checked: Any = None
            mc = re.match(r"^\[([ xX])\]\s+(.*)$", content)
            if mc:
                checked = mc.group(1).lower() == "x"
                content = mc.group(2).strip()

            # 保留行内图片语法，供 markdown_to_mubu_tree 提取为 images 字段
            content = content

            node: Dict[str, Any] = {"id": next_id(), "text": content, "children": []}
            if checked is not None:
                node["checked"] = checked

            while stack and stack[-1][0] >= depth:
                stack.pop()
            parent = stack[-1][1] if stack else root_node
            parent.setdefault("children", []).append(node)
            stack.append((depth, node))

    return {"node": root_node}


# ---------------------------------------------------------------------------
# Node normalisation (for changeset payloads)
# ---------------------------------------------------------------------------


def normalize_node(node: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively fill missing contract fields on a node dict."""
    if not isinstance(node, dict):
        return node
    now = int(time.time() * 1000)
    node.setdefault("note", "")
    node.setdefault("collapsed", False)
    node.setdefault("finish", False)
    node.setdefault("priority", 0)
    node.setdefault("color", 0)
    node.setdefault("createTime", node.get("timestamp") or now)
    node.setdefault("modifyTime", node.get("timestamp") or now)
    node.setdefault("timestamp", now)
    for child in node.get("children") or []:
        normalize_node(child)
    return node


# ---------------------------------------------------------------------------
# OPML export
# ---------------------------------------------------------------------------


def doc_to_opml(doc: Dict[str, Any]) -> str:
    """Convert a Mubu document to OPML 2.0 XML."""

    def build(node: Dict[str, Any], parent: ET.Element) -> None:
        text = (node.get("text") or "").replace("\n", " ")
        outline = ET.SubElement(parent, "outline", text=text)
        note = node.get("note")
        if note:
            outline.set("_note", note)
        images = node.get("images") or []
        if images:
            uri = (images[0].get("uri") or "")
            if uri:
                outline.set("_image", uri)
        for child in node.get("children") or []:
            build(child, outline)

    opml = ET.Element("opml", version="2.0")
    head = ET.SubElement(opml, "head")
    nodes = doc.get("nodes")
    if nodes:
        title = (nodes[0].get("text") or doc.get("name") or "mubu-export").replace("\n", " ")
        ET.SubElement(head, "title").text = title
        body = ET.SubElement(opml, "body")
        for node in nodes:
            build(node, body)
    else:
        root = doc.get("node") or doc
        title = (root.get("text") or "mubu-export").replace("\n", " ")
        ET.SubElement(head, "title").text = title
        body = ET.SubElement(opml, "body")
        build(root, body)

    ET.indent(opml, space="  ")
    return ET.tostring(opml, encoding="utf-8", xml_declaration=True).decode("utf-8")


# ---------------------------------------------------------------------------
# FreeMind (Freeplane) export
# ---------------------------------------------------------------------------


def doc_to_freeplane(doc: Dict[str, Any]) -> str:
    """Convert a Mubu document to FreeMind XML."""

    def build(node: Dict[str, Any], parent: ET.Element) -> None:
        for child in node.get("children") or []:
            text = (child.get("text") or "").replace("\n", " ")
            node_el = ET.SubElement(parent, "node", text=text)
            images = child.get("images") or []
            if images:
                uri = (images[0].get("uri") or "")
                if uri:
                    node_el.set("image", uri)
            note = child.get("note")
            if note:
                note_el = ET.SubElement(node_el, "richcontent", type="note")
                ET.SubElement(note_el, "html").text = note
            build(child, node_el)

    mindmap = ET.Element("map", version="1.0.1")
    nodes = doc.get("nodes")
    if nodes:
        title = (nodes[0].get("text") or doc.get("name") or "mubu-export").replace("\n", " ")
        root_node = ET.SubElement(mindmap, "node", text=title)
        build(nodes[0], root_node)
        for extra in nodes[1:]:
            text = (extra.get("text") or "mubu-export").replace("\n", " ")
            extra_el = ET.SubElement(root_node, "node", text=text)
            note = extra.get("note")
            if note:
                note_el = ET.SubElement(extra_el, "richcontent", type="note")
                ET.SubElement(note_el, "html").text = note
            build(extra, extra_el)
    else:
        root = doc.get("node") or doc
        title = (root.get("text") or "mubu-export").replace("\n", " ")
        root_node = ET.SubElement(mindmap, "node", text=title)
        build(root, root_node)

    ET.indent(mindmap, space="  ")
    return ET.tostring(mindmap, encoding="utf-8", xml_declaration=True).decode("utf-8")


# ---------------------------------------------------------------------------
# Filename safety
# ---------------------------------------------------------------------------

_BAD_CHARS = ['/', ':', '*', '?', '"', '<', '>', '|', '\\']


def safe_filename(name: str) -> str:
    """Sanitise a name for use as a file name."""
    cleaned = ''.join('_' if ch in _BAD_CHARS else ch for ch in name).strip()
    return cleaned or 'untitled'
