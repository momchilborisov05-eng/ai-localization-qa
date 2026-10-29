"""
XLIFF 1.2 and XLIFF 2.0 reader/writer (Phrase TMS exports both; .mxliff is XLIFF 1.2 with extras).

The file is edited in place with lxml, so everything this tool does not touch (headers, notes,
alt-trans, Phrase metadata, skeletons) is written back unchanged. Inline elements are masked as
⟦n⟧ tokens and restored byte-for-byte from the original markup.
"""

from __future__ import annotations

import re
from pathlib import Path

from lxml import etree

from .errors import LocQAError
from .models import Document, Segment
from .tags import unmask_to_xml

NS12 = "urn:oasis:names:tc:xliff:document:1.2"
NS20 = "urn:oasis:names:tc:xliff:document:2.0"
NS_SLR = "urn:oasis:names:tc:xliff:sizerestriction:2.0"
NS_MEMSOURCE = "http://www.memsource.com/mxlf/2.0"
PAIRED = {"g", "pc", "mrk"}  # elements that wrap translatable text
_XMLNS_RE = re.compile(r'\s+xmlns(?::[\w.-]+)?="[^"]*"')


def _local(el) -> str:
    return etree.QName(el).localname


def _raw(el) -> str:
    """Serialize an inline element (without its tail), dropping namespace declarations."""
    return _XMLNS_RE.sub("", etree.tostring(el, with_tail=False, encoding="unicode"))


class _Masker:
    """Turns the children of a <source>/<target> element into ⟦n⟧ tokens.

    With a reference (the source's token map), the same inline markup in a target gets the
    same token number, so source and target can be compared token by token.
    """

    def __init__(self, reference: dict[str, str] | None = None):
        self.tags: dict[str, str] = {}
        self.reference = reference or {}
        self._used: set[str] = set()
        numbers = [int(t.strip("⟦/⟧")) for t in self.reference]
        self.next_id = max(numbers, default=0) + 1

    def _number(self, raw: str) -> str:
        for tok, ref_raw in self.reference.items():
            if not tok.startswith("⟦/") and ref_raw == raw and tok not in self._used:
                self._used.add(tok)
                return tok[1:-1]
        n = str(self.next_id)
        self.next_id += 1
        return n

    def mask(self, el) -> str:
        parts = [el.text or ""]
        for child in el:
            if not isinstance(child.tag, str):  # comments, processing instructions
                parts.append(child.tail or "")
                continue
            raw = _raw(child)
            if _local(child) in PAIRED:
                start = raw[: raw.index(">") + 1]
                if start.endswith("/>"):
                    start = start[:-2].rstrip() + ">"
                name = re.match(r"<([^\s>/]+)", start).group(1)
                n = self._number(start)
                self.tags[f"⟦{n}⟧"] = start
                self.tags[f"⟦/{n}⟧"] = f"</{name}>"
                parts += [f"⟦{n}⟧", self.mask(child), f"⟦/{n}⟧"]
            else:
                n = self._number(raw)
                self.tags[f"⟦{n}⟧"] = raw
                parts.append(f"⟦{n}⟧")
            parts.append(child.tail or "")
        return "".join(parts)


def _text_of(el) -> str:
    return "".join(el.itertext()).strip() if el is not None else ""


class XliffDocument(Document):
    def __init__(self, path: Path, tree, version: str, source_lang: str, target_lang: str,
                 segments: list[Segment], handles: dict):
        super().__init__(path, source_lang, target_lang, segments)
        self.tree = tree
        self.version = version
        self.format_name = f"xliff{version.replace('.', '')}"
        self._handles = handles  # segment id -> element that owns <source>/<target>

    # ---------- reading ----------

    @classmethod
    def load(cls, path: Path) -> "XliffDocument":
        parser = etree.XMLParser(resolve_entities=False, no_network=True, remove_comments=False)
        try:
            tree = etree.parse(str(path), parser)
        except etree.XMLSyntaxError as e:
            raise LocQAError(f"{path} is not valid XML: {e}") from e
        root = tree.getroot()
        if root.tag == f"{{{NS12}}}xliff":
            return cls._load12(path, tree)
        if root.tag == f"{{{NS20}}}xliff":
            return cls._load20(path, tree)
        raise LocQAError(f"{path} is not XLIFF 1.2 or 2.0 (root element is {root.tag}).")

    @classmethod
    def _load12(cls, path, tree) -> "XliffDocument":
        q = lambda name: f"{{{NS12}}}{name}"  # noqa: E731
        root = tree.getroot()
        files = root.findall(q("file"))
        if not files:
            raise LocQAError(f"{path} has no <file> element.")
        source_lang = files[0].get("source-language", "en")
        target_lang = files[0].get("target-language", "")
        segments, handles = [], {}
        for f_index, file_el in enumerate(files):
            for tu in file_el.iter(q("trans-unit")):
                if tu.get("translate") == "no" or tu.get(f"{{{NS_MEMSOURCE}}}locked") == "true":
                    continue
                source_el = tu.find(q("source"))
                if source_el is None:
                    continue
                seg_id = tu.get("id") or f"tu{len(segments) + 1}"
                if seg_id in handles:
                    seg_id = f"{f_index + 1}:{seg_id}"
                masker = _Masker()
                source = masker.mask(source_el)
                target_el = tu.find(q("target"))
                target = _Masker(masker.tags).mask(target_el) if target_el is not None else ""
                max_chars = None
                if tu.get("maxwidth") and tu.get("size-unit", "char") == "char":
                    max_chars = int(tu.get("maxwidth"))
                note = " ".join(_text_of(n) for n in tu.findall(q("note")))
                segments.append(Segment(seg_id, source, target, masker.tags, note, max_chars))
                handles[seg_id] = tu
        return cls(path, tree, "1.2", source_lang, target_lang, segments, handles)

    @classmethod
    def _load20(cls, path, tree) -> "XliffDocument":
        q = lambda name: f"{{{NS20}}}{name}"  # noqa: E731
        root = tree.getroot()
        source_lang = root.get("srcLang", "en")
        target_lang = root.get("trgLang", "")
        segments, handles = [], {}
        for unit in root.iter(q("unit")):
            if unit.get("translate") == "no":
                continue
            unit_segments = unit.findall(q("segment"))
            notes_el = unit.find(q("notes"))
            note = " ".join(_text_of(n) for n in notes_el.findall(q("note"))) if notes_el is not None else ""
            unit_limit = unit.get(f"{{{NS_SLR}}}sizeRestriction")
            for index, seg_el in enumerate(unit_segments, start=1):
                source_el = seg_el.find(q("source"))
                if source_el is None:
                    continue
                unit_id = unit.get("id") or f"u{len(segments) + 1}"
                seg_id = unit_id if len(unit_segments) == 1 else f"{unit_id}/{seg_el.get('id') or index}"
                masker = _Masker()
                source = masker.mask(source_el)
                target_el = seg_el.find(q("target"))
                target = _Masker(masker.tags).mask(target_el) if target_el is not None else ""
                limit = seg_el.get(f"{{{NS_SLR}}}sizeRestriction") or unit_limit
                max_chars = int(limit) if limit and limit.isdigit() else None
                segments.append(Segment(seg_id, source, target, masker.tags, note, max_chars))
                handles[seg_id] = seg_el
        return cls(path, tree, "2.0", source_lang, target_lang, segments, handles)

    # ---------- writing ----------

    def save(self, path: Path, target_lang: str) -> list[str]:
        """Write the translated file. Returns ids of segments whose target could not be written."""
        root = self.tree.getroot()
        ns = NS12 if self.version == "1.2" else NS20
        nsdecl = " ".join(
            f'xmlns="{uri}"' if prefix is None else f'xmlns:{prefix}="{uri}"'
            for prefix, uri in root.nsmap.items()
        )
        if self.version == "1.2":
            for file_el in root.findall(f"{{{ns}}}file"):
                file_el.set("target-language", target_lang)
        else:
            root.set("trgLang", target_lang)

        failed = []
        for seg in self.segments:
            owner = self._handles[seg.id]
            target_el = owner.find(f"{{{ns}}}target")
            if target_el is None:
                source_el = owner.find(f"{{{ns}}}source")
                index = owner.index(source_el)
                indent = owner.text if index == 0 else owner[index - 1].tail
                target_el = etree.Element(f"{{{ns}}}target")
                target_el.tail = source_el.tail
                source_el.tail = indent
                owner.insert(index + 1, target_el)
            ok = self._fill(target_el, seg, nsdecl)
            if not ok:
                failed.append(seg.id)
            self._set_state(owner, target_el, seg, ok)

        etree.cleanup_namespaces(root)
        body = etree.tostring(self.tree, encoding="UTF-8", xml_declaration=False).decode("utf-8")
        body = re.sub(r"-->(?=<)", "-->\n", body)  # lxml drops line breaks between top-level nodes
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + body + "\n", encoding="utf-8", newline="")
        return failed

    @staticmethod
    def _fill(target_el, seg: Segment, nsdecl: str) -> bool:
        tail = target_el.tail
        for child in list(target_el):
            target_el.remove(child)
        target_el.text = None
        target_el.tail = tail
        try:
            xml = unmask_to_xml(seg.target, seg.tags)
            wrapper = etree.fromstring(f"<wrapper {nsdecl}>{xml}</wrapper>")
        except (ValueError, etree.XMLSyntaxError):
            return False  # broken tags: leave the target empty rather than write invalid XML
        target_el.text = wrapper.text
        for child in list(wrapper):
            target_el.append(child)
        return True

    def _set_state(self, owner, target_el, seg: Segment, ok: bool) -> None:
        if self.version == "1.2":
            if not ok or not seg.target:
                target_el.set("state", "needs-translation")
            elif seg.needs_review:
                target_el.set("state", "needs-review-translation")
            else:
                target_el.set("state", "translated")
        else:
            owner.set("state", "translated" if ok and seg.target else "initial")
            if seg.needs_review:
                owner.set("subState", "locqa:needs-review")
            elif "subState" in owner.attrib:
                del owner.attrib["subState"]
