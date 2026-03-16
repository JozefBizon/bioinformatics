import numpy as np
from Bio.Align import substitution_matrices  # biopython
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BLOSUM62 = substitution_matrices.load("BLOSUM62")
PAM250 = substitution_matrices.load("PAM250")

DNA_MATRIX = {
    ("A", "A"): 5,
    ("A", "T"): -4,
    ("A", "G"): -4,
    ("A", "C"): -4,
    ("T", "A"): -4,
    ("T", "T"): 5,
    ("T", "G"): -4,
    ("T", "C"): -4,
    ("G", "A"): -4,
    ("G", "T"): -4,
    ("G", "G"): 5,
    ("G", "C"): -4,
    ("C", "A"): -4,
    ("C", "T"): -4,
    ("C", "G"): -4,
    ("C", "C"): 5,
}


def match_symbol(a, b, score_matrix):
    if a == b:
        return "|"
    try:
        s = float(score_matrix[a, b])
    except (KeyError, TypeError):
        s = float(score_matrix.get((a, b), score_matrix.get((b, a), -1)))
    if s > 0:
        return ":"
    else:
        return "."


def debug_match_line(aligned1, aligned2, match_line, score_matrix):
    lines = []
    lines.append("\n--- DEBUG match line ---")
    lines.append(f"{'Pos':>4} {'A':>3} {'B':>3} {'Score':>6} {'Symbol':>6}")
    lines.append("-" * 30)
    idx = 0
    for a, b, sym in zip(aligned1, aligned2, match_line):
        if a == "-" or b == "-":
            lines.append(f"{idx:>4} {a:>3} {b:>3} {'GAP':>6} {sym!r:>6}")
        else:
            try:
                s = float(score_matrix[a, b])
            except (KeyError, TypeError):
                s = float(score_matrix.get((a, b), score_matrix.get((b, a), -99)))
            expected = match_symbol(a, b, score_matrix)
            flag = " <-- ROZDIEL" if sym != expected else ""
            lines.append(f"{idx:>4} {a:>3} {b:>3} {s:>6.1f} {sym!r:>6}{flag}")
        idx += 1
    lines.append("--- END DEBUG ---\n")
    return "\n".join(lines)


def local_alignment(seq1, seq2, score_matrix, gap_open=10, gap_extend=0.5):
    max_score = 0
    max_pos = (0, 0)
    m, n = len(seq1), len(seq2)

    traceback = np.zeros((m + 1, n + 1), dtype=int)
    H = np.zeros((m + 1, n + 1))
    E = np.full((m + 1, n + 1), -np.inf)
    F = np.full((m + 1, n + 1), -np.inf)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            try:
                s = score_matrix[seq1[i - 1], seq2[j - 1]]
            except (KeyError, TypeError):
                s = score_matrix.get(
                    (seq1[i - 1], seq2[j - 1]),
                    score_matrix.get((seq2[j - 1], seq1[i - 1]), -4),
                )

            E[i][j] = max(H[i][j - 1] - gap_open, E[i][j - 1] - gap_extend)
            F[i][j] = max(H[i - 1][j] - gap_open, F[i - 1][j] - gap_extend)
            diag = H[i - 1][j - 1] + s

            H[i][j] = max(0, diag, E[i][j], F[i][j])

            if H[i][j] == 0:
                traceback[i][j] = 0
            elif H[i][j] == diag:
                traceback[i][j] = 1
            elif H[i][j] == F[i][j]:
                traceback[i][j] = 2
            else:
                traceback[i][j] = 3

            if H[i][j] >= max_score:
                max_score = H[i][j]
                max_pos = (i, j)

    aligned1, aligned2, matches = [], [], []
    i, j = max_pos

    while H[i][j] > 0:
        if traceback[i][j] == 1:
            aligned1.append(seq1[i - 1])
            aligned2.append(seq2[j - 1])
            matches.append(match_symbol(seq1[i - 1], seq2[j - 1], score_matrix))
            i, j = i - 1, j - 1
        elif traceback[i][j] == 2:
            aligned1.append(seq1[i - 1])
            aligned2.append("-")
            matches.append(" ")
            i -= 1
        elif traceback[i][j] == 3:
            aligned1.append("-")
            aligned2.append(seq2[j - 1])
            matches.append(" ")
            j -= 1
        else:
            break

    aligned1 = "".join(reversed(aligned1))
    aligned2 = "".join(reversed(aligned2))
    match_line = "".join(reversed(matches))

    return max_score, aligned1, aligned2, match_line, max_pos


def format_alignment(aligned1, aligned2, match_line, score, width=50):
    lines = []
    lines.append(f"Skóre: {score}\n")
    for start in range(0, len(aligned1), width):
        end = start + width
        lines.append(f"Seq1: {aligned1[start:end]}")
        lines.append(f"      {match_line[start:end]}")
        lines.append(f"Seq2: {aligned2[start:end]}\n")
    return "\n".join(lines)


def print_alignment(
    aligned1, aligned2, match_line, score, debug=False, score_matrix=None
):
    output = format_alignment(aligned1, aligned2, match_line, score)
    print(output)
    if debug and score_matrix is not None:
        print(debug_match_line(aligned1, aligned2, match_line, score_matrix))


def save_alignment(
    output_path,
    header,
    aligned1,
    aligned2,
    match_line,
    score,
    seq1_name,
    seq2_name,
    seq1_len,
    seq2_len,
    debug=False,
    score_matrix=None,
):
    content = []
    content.append("=" * 60)
    content.append(header)
    content.append("=" * 60)
    content.append(f"Sekvencia 1: {seq1_name} (dĺžka: {seq1_len})")
    content.append(f"Sekvencia 2: {seq2_name} (dĺžka: {seq2_len})")
    content.append("")
    content.append(format_alignment(aligned1, aligned2, match_line, score))
    if debug and score_matrix is not None:
        content.append(debug_match_line(aligned1, aligned2, match_line, score_matrix))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(content))
    print(f"[Uložené: {output_path}]")


def read_fasta(filepath):
    sequence = []
    header = ""
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                header = line[1:]
            else:
                sequence.append(line.upper())
    return header, "".join(sequence)


def run_dna_alignment(file1, file2, output_file="output_dna.txt"):
    title = "NUKLEOTIDOVÉ ZAROVNANIE (Smith-Waterman)"
    print("=" * 60)
    print(title)
    print("=" * 60)
    h1, seq1 = read_fasta(file1)
    h2, seq2 = read_fasta(file2)
    print(f"Sekvencia 1: {h1} (dĺžka: {len(seq1)})")
    print(f"Sekvencia 2: {h2} (dĺžka: {len(seq2)})\n")

    score, a1, a2, mline, _ = local_alignment(seq1, seq2, DNA_MATRIX)
    print_alignment(a1, a2, mline, score)
    save_alignment(
        os.path.join(BASE_DIR, output_file),
        title,
        a1,
        a2,
        mline,
        score,
        h1,
        h2,
        len(seq1),
        len(seq2),
    )


def run_protein_alignment(file1, file2, matrix_name="BLOSUM62", output_file=None):
    title = f"AMINOKYSELINOVÉ ZAROVNANIE – {matrix_name}"
    print("=" * 60)
    print(title)
    print("=" * 60)
    h1, seq1 = read_fasta(file1)
    h2, seq2 = read_fasta(file2)
    print(f"Sekvencia 1: {h1} (dĺžka: {len(seq1)})")
    print(f"Sekvencia 2: {h2} (dĺžka: {len(seq2)})\n")

    matrix = BLOSUM62 if matrix_name == "BLOSUM62" else PAM250
    score, a1, a2, mline, _ = local_alignment(seq1, seq2, matrix)
    print_alignment(a1, a2, mline, score)

    if output_file is None:
        output_file = f"output_protein_{matrix_name.lower()}.txt"
    save_alignment(
        os.path.join(BASE_DIR, output_file),
        title,
        a1,
        a2,
        mline,
        score,
        h1,
        h2,
        len(seq1),
        len(seq2),
    )


if __name__ == "__main__":
    run_dna_alignment(
        os.path.join(BASE_DIR, "BC092625.1.fasta"),
        os.path.join(BASE_DIR, "M25079.1.fasta"),
        output_file="output_dna.txt",
    )

    run_protein_alignment(
        os.path.join(BASE_DIR, "A0A096MK47.fasta"),
        os.path.join(BASE_DIR, "P68871.fasta"),
        "BLOSUM62",
        output_file="output_protein_blosum62.txt",
    )

    run_protein_alignment(
        os.path.join(BASE_DIR, "A0A096MK47.fasta"),
        os.path.join(BASE_DIR, "P68871.fasta"),
        "PAM250",
        output_file="output_protein_pam250.txt",
    )
