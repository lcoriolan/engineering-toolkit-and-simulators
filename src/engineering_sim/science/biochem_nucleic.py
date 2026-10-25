# DESCRIPTION: Biochemistry area — DNA/RNA tools: sequence analysis, primer design, restriction
#              maps and virtual gels, pairwise alignment, and mutation consequences (Biopython 1.88,
#              primer3-py 2.3.1). No protein-structure features (owned by the Proteins tab).
"""Nucleic-acid tools for the science toolkit Biochemistry area.

Imports Biopython and primer3-py lazily inside each function (this module is only ever
imported by ops_biochem handlers under configured science environment). Every reported quantity is a
``cite.value`` record: ``derived`` for exact computations (GC %, molecular weight,
reverse complement, translation), ``estimated`` for the nearest-neighbour extinction
coefficient and melting temperatures (model + parameter table), ``cited`` only where a
source was actually fetched during this build.

Honest limits (kept in every result's ``notes``):
* Molecular weight uses Biopython's average nucleotide monophosphate weights and a 5'-phosphate
  assumption; ambiguous bases are rejected there (they are fine for GC % / length).
* Extinction coefficient uses the nearest-neighbour table (Tataurov, You & Owczarzy 2008);
  double-stranded value is reported as ~2x single-stranded and does NOT model hypochromism.
* Tm uses the SantaLucia (1998) nearest-neighbour parameters via Biopython (Na=50 mM, dNTP=0,
  oligo correction saltcorr=5) and primer3's ``calc_tm`` at its defaults; neither is fetched here,
  so the parameter provenance is unverified while the computation is derived.
"""
from __future__ import annotations

from . import cite

# IUPAC nucleotide codes for validation (DNA and RNA). Ambiguity codes are allowed for
# length/GC% but rejected by molecular-weight and translation where a single residue is unknown.
_IUPAC_DNA = set("GATCRYWSMKHBVDN")
_IUPAC_RNA = set("GAUCRYWSMKHBVDN")
_ACGT = set("ACGT")
_ACGU = set("ACGU")

_MAX_ANALYZE = 50000   # nt — bound the expensive steps
_MAX_ALIGN = 5000      # nt per sequence — pairwise aligner is O(n*m)

# A curated "common" set of type-II enzymes for the default digest (Bio.Restriction's CommOnly
# batch is far too large — 600+ entries including degenerate cutters). These are the standard
# 6-cutter (and a few 8-cutter) enzymes in everyday cloning work.
_COMMON_ENZYMES = ["EcoRI", "EcoRV", "BamHI", "BglII", "HindIII", "XhoI", "XbaI", "SpeI",
                   "NotI", "NdeI", "NcoI", "PstI", "SalI", "SacI", "SacII", "SmaI", "XmaI",
                   "KpnI", "ApaI", "PvuII", "ClaI", "SphI", "EagI", "HincII", "HpaI", "NheI",
                   "MluI", "AscI", "FseI"]

# Nearest-neighbour extinction coefficients at 260 nm (L mol^-1 cm^-1), Tataurov, You &
# Owczarzy, Biophys. Chem. 133, 66-70 (2008); the same values are used by IDT's OligoAnalyzer.
_NN_EPS = {
    "AA": 27400, "AC": 21200, "AG": 25000, "AT": 22800,
    "CA": 21200, "CC": 14600, "CG": 18000, "CT": 15200,
    "GA": 25200, "GC": 17600, "GG": 21600, "GT": 20000,
    "TA": 23400, "TC": 16200, "TG": 19000, "TT": 16800,
}
_MONO_EPS = {"A": 15400, "C": 7400, "G": 11500, "T": 8700}


def _clean(seq):
    """Strip whitespace and uppercase; return the bare sequence string."""
    return "".join(str(seq).split()).upper()


def _detect_kind(seq, kind):
    """Resolve 'auto' to 'dna'/'rna' or validate an explicit kind."""
    if kind not in ("auto", "dna", "rna"):
        raise ValueError("kind must be 'auto', 'dna' or 'rna', got %r" % (kind,))
    if kind == "auto":
        has_u = "U" in seq
        has_t = "T" in seq
        if has_u and has_t:
            raise ValueError("Sequence mixes T and U; specify kind='dna' or 'rna' explicitly.")
        return "rna" if has_u else "dna"
    return kind


def _validate(seq, kind):
    """Validate the sequence against the IUPAC alphabet for its kind; raise with a clear reason."""
    alphabet = _IUPAC_RNA if kind == "rna" else _IUPAC_DNA
    bad = sorted(set(seq) - alphabet)
    if bad:
        raise ValueError(
            "Not a valid %s sequence: invalid characters %s. "
            "Use IUPAC codes (ACGT%s, ambiguity RYSWKMBDHVN%s)."
            % (kind.upper(), ",".join(bad), "U" if kind == "rna" else "",
               " plus U" if kind == "rna" else ""))
    if not seq:
        raise ValueError("Empty sequence.")


def _reverse_complement(seq):
    """Reverse complement via Biopython (exact; complement table includes ambiguity codes)."""
    from Bio.Seq import Seq
    return str(Seq(seq).reverse_complement())


def _gc_percent(seq):
    """GC % over the four canonical bases (ambiguity codes excluded from the denominator)."""
    gc = sum(1 for b in seq if b in ("G", "C"))
    atgc = sum(1 for b in seq if b in _ACGT or b == "U")
    if atgc == 0:
        return 0.0
    return 100.0 * gc / atgc


def _molecular_weights(seq, kind):
    """ss/ds molecular weight (g/mol) via Biopython; unambiguous sequences only."""
    from Bio.Seq import Seq
    if not (set(seq) <= (_ACGU if kind == "rna" else _ACGT)):
        return None  # caller adds a note
    dna = seq.replace("U", "T") if kind == "rna" else seq
    from Bio.SeqUtils import molecular_weight
    ss = molecular_weight(Seq(dna), seq_type="DNA")
    ds = molecular_weight(Seq(dna), seq_type="DNA", double_stranded=True)
    return ss, ds


def _extinction(seq):
    """Nearest-neighbour epsilon260 (L mol^-1 cm^-1) for a single strand (U treated as T)."""
    dna = seq.replace("U", "T")
    if not dna or not (set(dna) <= _ACGT):
        return None
    total = sum(_NN_EPS.get(dna[i:i + 2], 0) for i in range(len(dna) - 1))
    if len(dna) == 1:
        return float(_MONO_EPS[dna])
    # Subtract the interior monomers so each base is counted once, not once per flanking dimer.
    total -= sum(_MONO_EPS[b] for b in dna[1:-1])
    return float(total)


def _tm_nn(seq, kind):
    """Nearest-neighbour Tm (deg C) via Biopython Tm_NN; DNA table for DNA, DNA table on U->T for RNA."""
    from Bio.Seq import Seq
    from Bio.SeqUtils import MeltingTemp
    if not (set(seq) <= (_ACGU if kind == "rna" else _ACGT)):
        return None
    dna = seq.replace("U", "T") if kind == "rna" else seq
    # SantaLucia (1998) unified parameters (Biopython default DNA_NN3); Na=50 mM, dNTP=0,
    # oligo correction saltcorr=5. dnac1/dnac2 left at Biopython's 25 nM default.
    return float(MeltingTemp.Tm_NN(Seq(dna), Na=50, K=0, Tris=0, Mg=0, dNTPs=0, saltcorr=5))


def _tm_primer3(seq):
    """primer3 calc_tm (deg C) at its defaults (Breslauer NN); DNA only."""
    import primer3
    if not (set(seq) <= _ACGT):
        return None
    return float(primer3.calc_tm(seq))


def _transcribe(seq, kind):
    """DNA -> RNA transcript (T -> U); RNA passes through unchanged."""
    if kind == "rna":
        return seq
    return seq.replace("T", "U")


def _translate(seq):
    """Translate with the standard code, suppressing Biopython's partial-codon warning."""
    import warnings
    from Bio.Seq import Seq
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return str(Seq(seq).translate())


def _translate_frame(seq, offset):
    """Translate one frame (0-based offset) with the standard code; returns AA string (X for ambiguity)."""
    return _translate(seq[offset:])


def _orfs(seq, min_codons):
    """ORFs >= min_codons codons in all 6 frames (standard code). Returns a sorted list of dicts."""
    out = []
    strands = [(seq, +1), (_reverse_complement(seq), -1)]
    for strand_seq, strand in strands:
        for offset in range(3):
            codons = [strand_seq[i:i + 3] for i in range(offset, len(strand_seq) - 2, 3)]
            start = 0
            while start < len(codons):
                end = start
                run = []
                while end < len(codons) and len(codons[end]) == 3 and _aa_of(codons[end]) != "*":
                    run.append(codons[end])
                    end += 1
                if len(run) >= min_codons:
                    aa = _translate("".join(run))
                    nt_start = offset + start * 3
                    nt_end = offset + end * 3
                    out.append({
                        "strand": strand, "frame": offset + 1,
                        "start": nt_start, "end": nt_end,
                        "length_codons": len(run), "starts_with_met": run[0].upper() == "ATG",
                        "aa": aa,
                    })
                start = end + 1
    out.sort(key=lambda o: (-o["length_codons"], o["strand"], o["start"]))
    return out


def _aa_of(codon):
    """Amino acid for one codon (standard code); '*' for stop, 'X' for ambiguous/short."""
    if len(codon) != 3 or any(b not in _ACGT for b in codon):
        return "X"
    return _translate(codon)


def _codon_usage(seq):
    """Codon and amino-acid counts over a coding region (no partial final codon)."""
    codons = {}
    aas = {}
    for i in range(0, len(seq) - 2, 3):
        c = seq[i:i + 3]
        if any(b not in _ACGT for b in c):
            continue
        codons[c] = codons.get(c, 0) + 1
        aa = _translate(c)
        aas[aa] = aas.get(aa, 0) + 1
    return codons, aas


def analyze(seq, kind="auto"):
    """Full nucleic-acid analysis. Returns a dict of cite records plus the 6-frame ORF map."""
    raw = _clean(seq)
    if len(raw) > _MAX_ANALYZE:
        raise ValueError("Sequence too long (%d nt); cap is %d nt." % (len(raw), _MAX_ANALYZE))
    k = _detect_kind(raw, kind)
    _validate(raw, k)

    unambiguous = set(raw) <= (_ACGU if k == "rna" else _ACGT)
    result = {
        "sequence": raw,
        "kind": k,
        "length": cite.value(len(raw), "nt", "derived", method="count of residues"),
        "gc_percent": cite.value(_gc_percent(raw), "%", "derived",
                                 method="100 * (G+C) / (G+C+A+T%s)" % ("+U" if k == "rna" else "")),
        "reverse_complement": _reverse_complement(raw),
        "transcript": _transcribe(raw, k),
        "notes": [],
    }

    mw = _molecular_weights(raw, k)
    if mw is None:
        result["notes"].append("Molecular weight and extinction coefficient are skipped: "
                               "the sequence contains ambiguity codes (only A/C/G/T%s are weighted)."
                               % ("/U" if k == "rna" else ""))
    else:
        ss, ds = mw
        result["mw_ss"] = cite.value(ss, "g/mol", "derived",
                                     method="Biopython molecular_weight (average nucleotide monophosphate weights, 5'-phosphate assumed)")
        result["mw_ds"] = cite.value(ds, "g/mol", "derived",
                                     method="Biopython molecular_weight, double_stranded=True (complement strand added)")

    eps = _extinction(raw)
    if eps is not None:
        result["extinction_260"] = cite.value(eps, "L/(mol cm)", "estimated",
                                              method="nearest-neighbour sum (Tataurov, You & Owczarzy 2008 table)",
                                              note="single-stranded; ds ~2x but hypochromism (~10-20 %) is not modelled")
    if unambiguous:
        tm_nn = _tm_nn(raw, k)
        result["tm_nn"] = cite.value(tm_nn, "deg C", "estimated",
                                     method="Biopython Tm_NN, SantaLucia (1998) table",
                                     note="Na=50 mM, dNTP=0, oligo correction (saltcorr=5); table not fetched during build")
        tm_p3 = _tm_primer3(raw)
        result["tm_primer3"] = cite.value(tm_p3, "deg C", "estimated",
                                          method="primer3 calc_tm at defaults",
                                          note="Breslauer NN parameters; defaults not fetched during build")
    else:
        result["notes"].append("Melting temperatures are skipped for ambiguity-coded sequences.")

    # 6-frame translation (standard code, NCBI table 1) and ORFs >= 30 codons.
    frames = []
    for strand, strand_seq in ((+1, raw), (-1, _reverse_complement(raw))):
        for offset in range(3):
            aa = _translate_frame(strand_seq, offset)
            frames.append({
                "strand": strand, "frame": offset + 1,
                "label": ("+" if strand > 0 else "-") + str(offset + 1),
                "aa": aa[:1200], "aa_len": len(aa),
            })
    result["frames"] = frames
    result["orfs"] = _orfs(raw, 30)

    # Codon usage over the longest ORF (or frame +1 when no ORF of 30+ codons is present).
    coding = None
    if result["orfs"]:
        top = result["orfs"][0]
        coding = raw[top["start"]:top["end"]] if top["strand"] > 0 else _reverse_complement(raw)[top["start"]:top["end"]]
    else:
        coding = raw
    codons, aas = _codon_usage(coding)
    result["codon_usage"] = {"codons": {k: codons[k] for k in sorted(codons)},
                             "amino_acids": {k: aas[k] for k in sorted(aas)},
                             "region": "longest ORF" if result["orfs"] else "frame +1 (no ORF >= 30 codons)"}

    result["notes"].append("Translation uses the standard genetic code (NCBI table 1); ORFs are >= 30 codons "
                           "and are reported with their strand and reading frame.")
    return result


def primers(template, target_region=None, size=(18, 25), tm=(57, 63)):
    """Design primer pairs with primer3 against a template; report Tm, GC, product size, penalties,
    and hairpin/homodimer/heterodimer free energies from primer3's own secondary-structure calls."""
    import primer3
    raw = _clean(template)
    if len(raw) > 100000:
        raise ValueError("Template too long (%d nt); cap is 100000 nt." % (len(raw),))
    _validate(raw, "dna")

    lo, hi = size
    tmlo, tmhi = tm
    if not (1 <= lo <= hi <= 60):
        raise ValueError("Primer size must be within 1..60 and lo <= hi.")
    if not (0 <= tmlo <= tmhi <= 120):
        raise ValueError("Primer Tm must be within 0..120 and lo <= hi.")

    seq_args = {"SEQUENCE_ID": "template", "SEQUENCE_TEMPLATE": raw}
    if target_region:
        start, end = target_region
        if start < 0 or end > len(raw) or start >= end:
            raise ValueError("target_region (%r) is out of range for a %d nt template." % (target_region, len(raw)))
        seq_args["SEQUENCE_TARGET"] = [int(start), max(0, int(end - start))]
    global_args = {
        "PRIMER_OPT_SIZE": int((lo + hi) / 2),
        "PRIMER_MIN_SIZE": int(lo), "PRIMER_MAX_SIZE": int(hi),
        "PRIMER_OPT_TM": (tmlo + tmhi) / 2.0, "PRIMER_MIN_TM": float(tmlo), "PRIMER_MAX_TM": float(tmhi),
        "PRIMER_NUM_RETURN": 5, "PRIMER_PRODUCT_SIZE_RANGE": [[int(lo * 2), 1000]],
    }
    design = primer3.design_primers(seq_args, global_args)
    pairs = []
    for i in range(int(design.get("PRIMER_PAIR_NUM_RETURNED", 0))):
        lp = "PRIMER_LEFT_%d" % i
        rp = "PRIMER_RIGHT_%d" % i
        left = design.get(lp + "_SEQUENCE")
        right = design.get(rp + "_SEQUENCE")
        if not left or not right:
            continue
        hairpin_l = primer3.calc_hairpin(left)
        homodim_l = primer3.calc_homodimer(left)
        hairpin_r = primer3.calc_hairpin(right)
        homodim_r = primer3.calc_homodimer(right)
        hetero = primer3.calc_heterodimer(left, right)
        pairs.append({
            "left": left, "right": right,
            "left_tm": design.get(lp + "_TM"), "right_tm": design.get(rp + "_TM"),
            "left_gc": design.get(lp + "_GC_PERCENT"), "right_gc": design.get(rp + "_GC_PERCENT"),
            "product_size": design.get("PRIMER_PAIR_%d_PRODUCT_SIZE" % i),
            "left_penalty": design.get(lp + "_PENALTY"), "right_penalty": design.get(rp + "_PENALTY"),
            "pair_penalty": design.get("PRIMER_PAIR_%d_PENALTY" % i),
            "dg": {
                "left_hairpin_kcal": hairpin_l.dg / 1000.0, "left_homodimer_kcal": homodim_l.dg / 1000.0,
                "right_hairpin_kcal": hairpin_r.dg / 1000.0, "right_homodimer_kcal": homodim_r.dg / 1000.0,
                "heterodimer_kcal": hetero.dg / 1000.0,
            },
            "warnings": _dimer_warnings(hairpin_l, homodim_l, hairpin_r, homodim_r, hetero),
        })
    result = {"template_length": len(raw), "pairs": pairs,
              "notes": ["Primer3 (Untergasser et al., Nucleic Acids Res. 40, e115 (2012)) at the stated size/Tm ranges. "
                        "Secondary-structure free energies are primer3's calc_hairpin/calc_homodimer/calc_heterodimer (kcal/mol); "
                        "these are computations, not fetched citations."]}
    return result


def _dimer_warnings(*thermo):
    """Warn when any self/hetero dimer is stably folded (|dG| > 6 kcal/mol, a common rule of thumb)."""
    w = []
    names = ["left hairpin", "left homodimer", "right hairpin", "right homodimer", "heterodimer"]
    for name, t in zip(names, thermo):
        if t.structure_found and abs(t.dg) / 1000.0 > 6.0:
            w.append("%s: dG = %.1f kcal/mol (stable)" % (name, t.dg / 1000.0))
    return w


def restriction(seq, enzymes="common"):
    """Restriction sites and fragment sizes via Bio.Restriction; return a site list and a
    virtual-gel fragment table (log-size positions for the page)."""
    from Bio.Seq import Seq
    from Bio.Restriction import Analysis, RestrictionBatch
    from Bio import Restriction as _R
    raw = _clean(seq)
    if len(raw) > 20000:
        raise ValueError("Sequence too long (%d nt); cap is 20000 nt." % (len(raw),))
    _validate(raw, "dna")

    if enzymes == "common":
        batch = RestrictionBatch([getattr(_R, n) for n in _COMMON_ENZYMES if hasattr(_R, n)])
    elif isinstance(enzymes, str):
        batch = RestrictionBatch([enzymes])
    else:
        batch = RestrictionBatch(list(enzymes))

    analysis = Analysis(batch, Seq(raw))
    full = analysis.full()
    sites = []
    fragments_by_enzyme = {}
    for enzyme in sorted(full.keys(), key=lambda e: str(e)):
        cuts = sorted(full[enzyme])
        frags = _fragments(cuts, len(raw))
        fragments_by_enzyme[str(enzyme)] = frags
        sites.append({"enzyme": str(enzyme), "site": enzyme.site, "cuts": cuts, "fragments": frags})

    result = {
        "length": len(raw),
        "sites": sites,
        "notes": ["Cut positions are 0-based coordinates as reported by Bio.Restriction. "
                  "Fragments include the flanking ends (a linear sequence has one more fragment than cuts)."],
    }
    return result


def _fragments(cuts, length):
    """Fragment sizes (bp) from sorted cut positions, descending (gel-like order)."""
    if not cuts:
        return [length]
    points = cuts + [length]
    sizes = [points[0]] + [points[i] - points[i - 1] for i in range(1, len(points))]
    return sorted(sizes, reverse=True)


def align(seq_a, seq_b, mode="global"):
    """Pairwise alignment with stated scoring; identity % and a formatted (capped) alignment."""
    from Bio import Align
    from Bio.Seq import Seq
    a, b = _clean(seq_a), _clean(seq_b)
    if len(a) > _MAX_ALIGN or len(b) > _MAX_ALIGN:
        raise ValueError("Alignment is capped at %d nt per sequence." % (_MAX_ALIGN,))
    if not a or not b:
        raise ValueError("Both sequences must be non-empty.")
    if mode not in ("global", "local"):
        raise ValueError("mode must be 'global' or 'local'.")

    aligner = Align.PairwiseAligner()
    aligner.mode = mode
    # Smith-Waterman-style integer scoring: +2 match, -1 mismatch, -2 gap open, -0.5 gap extend.
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -0.5
    aligner.target_internal_open_gap_score = -2
    aligner.query_internal_open_gap_score = -2

    alignments = aligner.align(Seq(a), Seq(b))
    best = alignments[0]
    aln_str = str(best)
    score = best.score
    # Identity over aligned columns (ignoring gaps).
    lines = aln_str.split("\n")
    matches = sum(1 for i in range(len(lines[0])) if lines[0][i] == lines[2][i])
    columns = sum(1 for i in range(len(lines[0])) if lines[0][i] != "-" and lines[2][i] != "-")
    identity = 100.0 * matches / columns if columns else 0.0
    return {
        "mode": mode,
        "score": score,
        "identity_percent": cite.value(identity, "%", "derived",
                                       method="100 * identical aligned residues / aligned columns (gaps excluded)"),
        "len_a": len(a), "len_b": len(b),
        "alignment": _format_alignment(aln_str, cap=5000),
        "notes": ["Scoring: match +2, mismatch -1, gap open -2, gap extend -0.5 (Bio.Align.PairwiseAligner). "
                  "Alignment display is capped at 5000 columns."],
    }


def _format_alignment(aln_str, cap):
    """Wrap a three-line alignment text into <= cap columns."""
    lines = aln_str.split("\n")
    if len(lines) != 3 or len(lines[0]) <= cap:
        return aln_str
    out = []
    for i in range(0, len(lines[0]), cap):
        out.append("%s\n%s\n%s" % (lines[0][i:i + cap], lines[1][i:i + cap], lines[2][i:i + cap]))
    return "\n\n".join(out)


def mutate(seq, spec):
    """Apply point substitutions/insertions/deletions and report the translated consequence."""
    raw = _clean(seq)
    if len(raw) > _MAX_ANALYZE:
        raise ValueError("Sequence too long (%d nt); cap is %d nt." % (len(raw), _MAX_ANALYZE))
    _validate(raw, "dna")
    specs = spec if isinstance(spec, list) else [spec]
    mut = list(raw)
    changes = []
    for op in specs:
        if not isinstance(op, dict):
            raise ValueError("Each mutation must be a dict, e.g. {'type':'sub','pos':3,'base':'C'}.")
        t = op.get("type")
        pos = int(op.get("pos", 0))
        if t == "sub":
            base = str(op.get("base", "")).upper()
            if len(base) != 1 or base not in _IUPAC_DNA:
                raise ValueError("sub needs a single IUPAC base, got %r" % (op.get("base"),))
            if not (0 <= pos < len(mut)):
                raise ValueError("sub position %d out of range." % pos)
            changes.append({"type": "sub", "pos": pos, "from": mut[pos], "to": base})
            mut[pos] = base
        elif t == "ins":
            ins = _clean(op.get("seq", ""))
            _validate(ins, "dna")
            if not (0 <= pos <= len(mut)):
                raise ValueError("ins position %d out of range." % pos)
            changes.append({"type": "ins", "pos": pos, "seq": ins})
            mut[pos:pos] = list(ins)
        elif t == "del":
            n = int(op.get("n", 1))
            if n <= 0 or pos < 0 or pos + n > len(mut):
                raise ValueError("del position/count out of range.")
            changes.append({"type": "del", "pos": pos, "seq": "".join(mut[pos:pos + n])})
            del mut[pos:pos + n]
        else:
            raise ValueError("Unknown mutation type %r (use sub/ins/del)." % (t,))

    mutated = "".join(mut)
    # Translated consequence in frame +1 (the default frame); note the assumption.
    aa_orig = _translate(raw)
    aa_new = _translate(mutated)
    aa_changes = []
    for i in range(max(len(aa_orig), len(aa_new))):
        o = aa_orig[i] if i < len(aa_orig) else "-"
        n = aa_new[i] if i < len(aa_new) else "-"
        if o != n:
            aa_changes.append({"index": i + 1, "from": o, "to": n})
    return {
        "original": raw,
        "mutated": mutated,
        "changes": changes,
        "aa_original": aa_orig[:2000],
        "aa_mutated": aa_new[:2000],
        "aa_changes": aa_changes[:200],
        "notes": ["Translation is reported in reading frame +1 only (the sequence is not re-phased); "
                  "frame-shifting insertions/deletions change the translation from the mutation site onward."],
    }
