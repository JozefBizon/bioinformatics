# Practicum 03: Alignment

## Introduction

In this practicum, we will focus on the alignment of sequencing reads to a reference genome or assembly.
You will learn how to use popular alignment tools such as BWA and Bowtie2, as well as how to process and analyze the resulting alignment files (SAM/BAM format) using Samtools.
This practicum will provide you with hands-on experience in performing read alignment, assessing alignment quality, and manipulating alignment files for downstream analyses.

## Objectives

By the end of this practicum, you will be able to:

1. Understand the structure and format of BAM/SAM files.

2. Understand the FASTA reference genome(s).

3. Index a reference genome for alignment.

4. Align sequencing reads to a reference genome using BWA and Bowtie2 for short reads, and Minimap2 for long reads.

5. Inspect and assess the quality of alignments using Samtools and IGV.

6. Compare different alignment tools and their outputs.

## Prerequisites

- Bioinformatics resources repository cloned to your local machine. If you haven't done this yet, follow the instructions in the main README file of the repository.

## Getting Started

Additional software/tools required for this practicum:

- [Samtools](http://www.htslib.org/)

- [IGV](https://software.broadinstitute.org/software/igv/)

- [BWA](http://bio-bwa.sourceforge.net/)

- [Bowtie2](http://bowtie-bio.sourceforge.net/bowtie2/index.shtml)

- [Minimap2](https://lh3.github.io/minimap2/)

You can install these tools using `conda/mamba`:

```bash

```

FASTQ files containing sequencing reads for alignment can be found in the `data/sequencing_reads/` directory of the bioinformatics resources repository.
You must download the reference genome FASTA reference file from the UCSC Biobank and prepare it for alignment.
The index construction should take a few minutes depending on your machine's performance.

```bash
# Download the hg38 reference genome from UCSC
wget https://hgdownload.gi.ucsc.edu/goldenPath/hg38/bigZips/latest/hg38.fa.gz
gunzip hg38.fa.gz

# Create contig dictionary and FASTA index
samtools faidx hg38.fa
samtools dict hg38.fa > hg38.dict

# Index the reference genome for BWA
bwa index hg38.fa

# Index the reference genome for Bowtie2
bowtie2-build hg38.fa hg38_bt2_index

# Index the reference genome for Minimap2
minimap2 -d hg38.mmi hg38.fa
```

## Further Reading and Resources

- [SAM/BAM Format Specification](https://samtools.github.io/hts-specs/SAMv1.pdf)

- [Samtools Documentation](http://www.htslib.org/doc/samtools.html)

- [IGV Documentation](https://software.broadinstitute.org/software/igv/documentation)

- [BWA Manual](http://bio-bwa.sourceforge.net/bwa.shtml)

- [Bowtie2 Manual](http://bowtie-bio.sourceforge.net/bowtie2/index.shtml)

- [Minimap2 Documentation](https://lh3.github.io/minimap2/minimap2.html)
